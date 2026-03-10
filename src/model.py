"""
model.py - Arquitectura CNN Multi-modal de DeepSolarEye v3.2

Versión mejorada de ImpactNet para predicción de soiling en paneles solares.
Input: Imágenes RGB (224×224) + Features ambientales (irradiance)
Output: Pérdida de potencia (regresión abierta, sin sigmoid)
"""

import torch
import torch.nn as nn

# Importar configuración de features ambientales
from src.config import (
    ENV_FC1_OUT,
    ENV_FC2_OUT,
    FUSION_INPUT,
    FUSION_OUTPUT,
    NUM_ENV_FEATURES,
)


class Net(nn.Module):
    """
    Red neuronal convolucional multi-modal basada en ImpactNet.
    
    CAMBIO CRÍTICO EN v3.0: Se removió sigmoid de la salida.
    CAMBIO v3.2: Arquitectura multi-modal (image + env features).
    
    Justificación v3.0:
    - En regresión, la salida debe ser LIBRE (sin restricciones [0,100]).
    - Si el modelo predice -50 o 150, es información diagnóstica valiosa.
    - Límites artificiales con sigmoid OCULTAN problemas de aprendizaje.
    - El post-procesing (clipping) es responsabilidad de la aplicación.
    
    Justificación v3.2 (multi-modal):
    - Inspirado en ImpactNet original que usa features ambientales.
    - La irradiance tiene correlación física con el soiling.
    - Rama MLP dedicada: Linear(1→32)→ReLU→Linear(32→64)→ReLU
    - Fusión tardía (late fusion): concat(image_96, env_64) → Linear(160,96)
    
    Arquitectura:
    - Rama visual: Conv2d(3→16, 7×7) + 5 AU + FC → 96 features
    - Rama ambiental: MLP(1→32→64) → 64 features
    - Fusión: concat → Linear(160→96) → 96 → 1
    
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
        # RAMA AMBIENTAL - MLP (v3.2)
        # ============================================================
        # Procesa features ambientales (irradiance) en paralelo a la CNN.
        # Inspirado en ImpactNet original: forward(au, ef)
        # Dimensiones: 1 → 32 → 64 (configurables en config.py)
        self.env_fc1 = nn.Linear(NUM_ENV_FEATURES, ENV_FC1_OUT)
        self.env_bn1 = nn.BatchNorm1d(ENV_FC1_OUT)
        self.env_fc2 = nn.Linear(ENV_FC1_OUT, ENV_FC2_OUT)
        self.env_bn2 = nn.BatchNorm1d(ENV_FC2_OUT)

        # ============================================================
        # CAPA DE FUSIÓN (v3.2)
        # ============================================================
        # Late fusion: concatena features visuales (96) + ambientales (64)
        # y las proyecta de vuelta a 96 dimensiones para la salida
        self.fusion = nn.Linear(FUSION_INPUT, FUSION_OUTPUT)  # 160 → 96

        # ============================================================
        # CAPA DE SALIDA (REGRESIÓN)
        # ============================================================
        # Final: 96 → 1 (salida única: predicción de power loss %)
        # SIN sigmoid: permite predicciones fuera [0, 100] como diagnóstico
        self.fc_final = nn.Linear(FUSION_OUTPUT, 1)

    def forward(self, x: torch.Tensor, env: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass de la red multi-modal.
        
        Arquitectura paso a paso:
        1. Rama visual: Convolución inicial + 5 AU + FC → 96 features
        2. Rama ambiental: MLP(env) → 64 features
        3. Fusión: concat(visual_96, env_64) → Linear(160, 96)
        4. Salida: 96 → 1 (regresión abierta)
        
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
        
        # 6. Rama ambiental: MLP sobre features ambientales (v3.2)
        ef = self.relu(self.env_bn1(self.env_fc1(env)))   # [B, 32]
        ef = self.relu(self.env_bn2(self.env_fc2(ef)))    # [B, 64]
        
        # 7. Fusión tardía (late fusion): concatenación + proyección
        # Patrón ImpactNet: torch.cat((ef, au), dim=1)
        fused = torch.cat((x, ef), dim=1)          # [B, 160]
        fused = self.relu(self.fusion(fused))       # [B, 96]
        
        # 8. Salida final - Regresión ABIERTA (sin sigmoid)
        output = self.fc_final(fused)
        
        return output


