"""
model.py - Arquitectura CNN con inyección directa de DeepSolarEye v3.3

Versión mejorada de ImpactNet para predicción de soiling en paneles solares.
Input: Imágenes RGB (224×224) + Features ambientales (irradiance)
Output: Pérdida de potencia (regresión abierta, sin sigmoid)
"""

import torch
import torch.nn as nn

# Importar configuración de features ambientales
from src.config import NUM_ENV_FEATURES


class Net(nn.Module):
    """
    Red neuronal convolucional con inyección directa basada en ImpactNet.
    
    CAMBIO CRÍTICO EN v3.0: Se removió sigmoid de la salida.
    CAMBIO v3.2: Rama MLP ambiental (1→32→64) + fusión (160→96).
    CAMBIO v3.3: Eliminada rama MLP. Inyección directa de irradiance.
    
    Justificación v3.0 (regresión abierta):
    - En regresión, la salida debe ser LIBRE (sin restricciones [0,100]).
    - Si el modelo predice -50 o 150, es información diagnóstica valiosa.
    - El post-procesing (clipping) es responsabilidad de la aplicación.
    
    Justificación v3.3 (inyección directa):
    - v3.2 con MLP(1→32→64) causó regresión: RMSE 9.62 vs 8.73 (v3.1).
    - Diagnóstico: 17,664 params extra para 1 feature (corr=0.10) → overfitting.
    - BN1d sobre 1 feature con batch=32 creó gap artificial train/val.
    - Solución: concat(cnn_96, irradiance_1) → Linear(97→1).
    - 1 parámetro extra vs 17,664 → mínimo riesgo de overfitting.
    
    Comparación de parámetros (rama ambiental):
    ┌──────────────┬────────────────────────────┬────────────┐
    │ Versión      │ Arquitectura               │ Params     │
    ├──────────────┼────────────────────────────┼────────────┤
    │ v3.1         │ Solo imagen                │ 0          │
    │ v3.2 (MLP)   │ 1→32→64 + fusión 160→96   │ 17,664     │
    │ v3.3 (direct)│ concat + Linear(97→1)      │ 1*         │
    └──────────────┴────────────────────────────┴────────────┘
    * 1 peso extra en fc_final (97 vs 96 entradas)
    
    Arquitectura v3.3:
    - Rama visual: Conv2d(3→16, 7×7) + 5 AU + FC → 96 features
    - Inyección:   concat(visual_96, irradiance_1) → 97 features
    - Salida:      Linear(97→1) → regresión abierta
    
    Input shape: [B, 3, 224, 224] + [B, 1] donde B = batch size
    Output shape: [B] predicciones de pérdida de potencia (%)
    """
    
    def __init__(self) -> None:
        """
        Inicialización de capas y módulos de la red.
        
        Nota: El número 384 en la primera capa FC se calcula como:
        96 canales × 2×2 spatial size (después de pooling progresivo).
        Verificar que coincida exactamente tras aplanado en forward().
        """
        super(Net, self).__init__()
        
        # ============================================================
        # BLOQUE DE EXTRACCIÓN DE CARACTERÍSTICAS INICIAL
        # ============================================================
        
        # Capa convolucional inicial: extrae características de bajo nivel
        # Input: [B, 3, 224, 224] → Output: [B, 16, 224, 224]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=7,
            padding=3
        )
        
        # BatchNorm después de conv1 (añadido en v3.1)
        # Justificación: Ioffe & Szegedy (2015) demuestran que BN:
        #   1. Reduce Internal Covariate Shift
        #   2. Permite LR más altos
        #   3. Actúa como regularizador adicional
        # Antes de v3.1, conv1 procesaba entrada sin normalización
        self.bn1 = nn.BatchNorm2d(16)
        
        # Average pooling para reducir resolución espacial sin perder información
        # kernel 3×3 reduce aprox. a 75×75 después de conv1
        self.pool = nn.AvgPool2d(kernel_size=3)
        
        # Dropout para regularización: previene overfitting en oversampling
        self.dropout = nn.Dropout(p=0.5)
        
        # ReLU activation (reutilizado en múltiples bloques)
        self.relu = nn.ReLU(inplace=True)

        # ============================================================
        # ANALYSIS UNIT 1 (AU1): 16→32 canales
        # ============================================================
        # Proyección: stride=2 reduce resolución espacial
        self.rcu1_conv = nn.Conv2d(16, 32, kernel_size=1, stride=2)
        # Bloque residual: Conv→BN→ReLU→Conv→BN
        self.rcu1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32)
        )
        
        # ============================================================
        # ANALYSIS UNIT 2 (AU2): 32→48 canales
        # ============================================================
        self.rcu2_conv = nn.Conv2d(32, 48, kernel_size=1, stride=2)
        self.rcu2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48)
        )
        
        # ============================================================
        # ANALYSIS UNIT 3 (AU3): 48→64 canales
        # ============================================================
        self.rcu3_conv = nn.Conv2d(48, 64, kernel_size=1, stride=2)
        self.rcu3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64)
        )

        # ============================================================
        # ANALYSIS UNIT 4 (AU4): 64→80 canales
        # ============================================================
        self.rcu4_conv = nn.Conv2d(64, 80, kernel_size=1, stride=2)
        self.rcu4 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=5, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 80, kernel_size=5, padding=2),
            nn.BatchNorm2d(80)
        )
        
        # ============================================================
        # ANALYSIS UNIT 5 (AU5): 80→96 canales
        # ============================================================
        self.rcu5_conv = nn.Conv2d(80, 96, kernel_size=1, stride=2)
        self.rcu5 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96)
        )

        # ============================================================
        # CAPAS TOTALMENTE CONECTADAS (CLASIFICADOR)
        # ============================================================
        # Primera FC: 384 entradas (96 canales × 2×2 spatial) → 96 neuronas
        # Nota: Verificar tamaño exacto tras pooling progresivo en forward()
        self.fu = nn.Linear(384, 96)
        
        # Segunda FC: 96 → 96 (bottleneck para regularización)
        self.fc0 = nn.Linear(96, 96)
        
        # ============================================================
        # CAPA DE SALIDA CON INYECCIÓN DIRECTA (v3.3)
        # ============================================================
        # v3.3: Eliminada rama MLP y capa de fusión.
        # Inyección directa: concat(visual_96, irradiance_1) → 97 → 1
        # Solo 1 parámetro extra vs 17,664 del MLP en v3.2.
        # SIN sigmoid: permite predicciones fuera [0, 100] como diagnóstico
        self.fc_final = nn.Linear(96 + NUM_ENV_FEATURES, 1)  # 97 → 1

    def forward(self, x: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con inyección directa de irradiance.
        
        Arquitectura paso a paso:
        1. Rama visual: Convolución inicial + 5 AU + FC → 96 features
        2. Inyección: concat(visual_96, irradiance_1) → 97 features
        3. Salida: Linear(97→1) → regresión abierta
        
        OPTIMIZACIÓN CRÍTICA:
        Para cada AU, se calcula la proyección UNA SOLA VEZ y se guarda
        en una variable temporal. Esto evita cómputo duplicado.
        
        Args:
            x (torch.Tensor): Batch de imágenes RGB normalizadas.
                             Shape: [B, 3, 224, 224]
                             Valores esperados: μ=0, σ=1 (ImageNet norm)
            env (torch.Tensor): Features ambientales (irradiance).
                               Shape: [B, NUM_ENV_FEATURES]
                               Valores esperados: [0, 1]
        
        Returns:
            torch.Tensor: Predicciones de pérdida de potencia.
                         Shape: [B, 1] (train.py aplica squeeze(dim=1) → [B])
                         Rango esperado: ~[0, 100]
                         Rango posible: [-∞, +∞] (diagnóstico sin restricción)
        
        Example:
            >>> model = Net()
            >>> x = torch.randn(32, 3, 224, 224)
            >>> env = torch.rand(32, 1)
            >>> y = model(x, env)
            >>> print(y.shape)
            torch.Size([32, 1])
        """
        
        # 1. Extracción inicial de características de bajo nivel
        # v3.1: Añadido BatchNorm después de conv1 para normalizar antes de AU1
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 2. Analysis Units con conexiones residuales
        # Patrón residual: x = relu(proj(x) + block(proj(x)))
        # Esta estructura permite gradientes más directos (ResNets)
        
        # AU1: proyección y residuo en una sola variable
        proj = self.rcu1_conv(x)
        x = self.relu(proj + self.rcu1(proj))
        
        # AU2
        proj = self.rcu2_conv(x)
        x = self.relu(proj + self.rcu2(proj))
        
        # AU3
        proj = self.rcu3_conv(x)
        x = self.relu(proj + self.rcu3(proj))

        # AU4
        proj = self.rcu4_conv(x)
        x = self.relu(proj + self.rcu4(proj))
        
        # AU5
        proj = self.rcu5_conv(x)
        x = self.relu(proj + self.rcu5(proj))
        
        # 3. Reduce dimensionalidad espacial (captura contexto global)
        x = self.pool(x)
        
        # 4. Aplanar: [B, C, H, W] → [B, C*H*W]
        # Ejemplo: [32, 96, 2, 2] → [32, 384]
        x = x.view(x.shape[0], -1)
        
        # 5. Capas totalmente conectadas con regularización (rama visual)
        x = self.relu(self.dropout(self.fu(x)))
        x = self.relu(self.dropout(self.fc0(x)))  # [B, 96]
        
        # 6. Inyección directa de irradiance (v3.3)
        # Concat simple: [B, 96] + [B, 1] → [B, 97]
        # Sin MLP intermedio: 1 param extra vs 17,664 en v3.2
        x = torch.cat((x, env), dim=1)  # [B, 97]
        
        # 7. Salida final - Regresión ABIERTA (sin sigmoid)
        output = self.fc_final(x)  # [B, 1]
        
        return output


