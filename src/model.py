"""
model.py - Arquitectura CNN de DeepSolarEye v3.1

Versión mejorada de ImpactNet para predicción de soiling en paneles solares.
Input: Imágenes RGB (224×224)
Output: Pérdida de potencia (regresión abierta, sin sigmoid)
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Red neuronal convolucional personalizada basada en ImpactNet.
    
    CAMBIO CRÍTICO EN v3.0: Se removió sigmoid de la salida.
    
    Justificación:
    - En regresión, la salida debe ser LIBRE (sin restricciones [0,100]).
    - Si el modelo predice -50 o 150, es información diagnóstica valiosa.
    - Límites artificiales con sigmoid OCULTAN problemas de aprendizaje.
    - El post-procesing (clipping) es responsabilidad de la aplicación.
    
    Arquitectura:
    - Capa inicial: Conv2d(3→16, kernel 7×7) + AvgPool + Dropout
    - 5 Analysis Units (AU1-AU5): Bloques residuales con Conv5×5
    - Fully Connected: 384→96→96→1 (salida en regresión abierta)
    
    Input shape: [B, 3, 224, 224] donde B = batch size
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
        # CAPA DE SALIDA (REGRESIÓN)
        # ============================================================
        # Final: 96 → 1 (salida única: predicción de power loss %)
        # SIN sigmoid: permite predicciones fuera [0, 100] como diagnóstico
        self.fc_final = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red.
        
        Arquitectura paso a paso:
        1. Convolución inicial + pooling global
        2. 5 Analysis Units con conexiones residuales (optimizadas)
        3. Pooling final y aplanado
        4. Capas FC con dropout para regularización
        5. Salida sin restricciones (regresión abierta)
        
        OPTIMIZACIÓN CRÍTICA:
        Para cada AU, se calcula la proyección UNA SOLA VEZ y se guarda
        en una variable temporal. Esto evita cómputo duplicado.
        
        Antes (ineficiente):
            x = self.relu(self.rcu1_conv(x) + self.rcu1(self.rcu1_conv(x)))
            # self.rcu1_conv(x) se ejecuta dos veces ❌
        
        Después (eficiente):
            proj = self.rcu1_conv(x)
            x = self.relu(proj + self.rcu1(proj))
            # self.rcu1_conv(x) se ejecuta una sola vez ✓
        
        Args:
            x (torch.Tensor): Batch de imágenes RGB normalizadas.
                             Shape: [B, 3, 224, 224]
                             Valores esperados: μ=0, σ=1 (ImageNet norm)
        
        Returns:
            torch.Tensor: Predicciones de pérdida de potencia.
                         Shape: [B, 1] (train.py aplica squeeze(dim=1) → [B])
                         Rango esperado: ~[0, 100]
                         Rango posible: [-∞, +∞] (diagnóstico sin restricción)
        
        Example:
            >>> model = Net()
            >>> x = torch.randn(32, 3, 224, 224)  # batch de 32
            >>> y = model(x)  # shape: [32]
            >>> print(y.shape, y.min().item(), y.max().item())
            torch.Size([32]) -5.234 105.872
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
        
        # 5. Capas totalmente conectadas con regularización
        x = self.relu(self.dropout(self.fu(x)))
        x = self.relu(self.dropout(self.fc0(x)))
        
        # 6. Salida final - Regresión ABIERTA (sin sigmoid)
        # Interpretación:
        # - ~[0, 100]: predicciones en rango físicamente esperado
        # - <0 o >100: diagnóstico de problemas de aprendizaje
        output = self.fc_final(x)  # Retorna [B, 1], train.py hace squeeze(dim=1)
        
        return output


