"""
Arquitectura CNN con bloques residuales para DeepSolarEye v4.0.

Implementa una red convolucional personalizada con 5 Unidades Convolucionales
Residales (RCU) progresivas y fusión multimodal de características visuales
y ambientales para regresión de pérdida de potencia en paneles solares.

"""


import torch
import torch.nn as nn
from src.config import NUM_ENV_FEATURES

class Net(nn.Module):

    def __init__(self) -> None:
        """Inicializa la arquitectura de la red neuronal."""
        super(Net, self).__init__()

        # Etapa inicial: extracción de características primarias
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7,
                               padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.AvgPool2d(kernel_size=3)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

        # Unidades Convolucionales Residuales (RCU)
        # Progresión de canales: 16->32->48->64->80->96 con reducción 2x por RCU
        self.rcu1_conv = nn.Conv2d(16, 32, kernel_size=1, stride=2)
        self.rcu1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32)
        )

        self.rcu2_conv = nn.Conv2d(32, 48, kernel_size=1, stride=2)
        self.rcu2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48)
        )

        self.rcu3_conv = nn.Conv2d(48, 64, kernel_size=1, stride=2)
        self.rcu3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64)
        )

        self.rcu4_conv = nn.Conv2d(64, 80, kernel_size=1, stride=2)
        self.rcu4 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=5, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 80, kernel_size=5, padding=2),
            nn.BatchNorm2d(80)
        )

        self.rcu5_conv = nn.Conv2d(80, 96, kernel_size=1, stride=2)
        self.rcu5 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96)
        )

        # Capas densas de la rama visual
        self.fu = nn.Linear(384, 96)
        self.fc0 = nn.Linear(96, 96)
        
        # Capa de salida: fusión multimodal visual + ambiental
        # Input: 96 (visual) + NUM_ENV_FEATURES (irradiancia) -> Output: 1 (pérdida %)
        self.fc_final = nn.Linear(96 + NUM_ENV_FEATURES, 1)

    def forward(self, x: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """Procesa imagen y features ambientales mediante forward pass multimodal."""
        # Etapa inicial: normalizar features de entrada
        x = self.relu(self.bn1(self.conv1(x)))

        # Unidades residuales: proyección + convoluciones + conexión residual
        # Estrategia: cada RCU incrementa canal y reduce espacial 2x
        # RCU 1: 16->32 canales (stride=2)
        proj = self.rcu1_conv(x)
        x = self.relu(proj + self.rcu1(proj))

        # RCU 2: 32->48 canales (stride=2)
        proj = self.rcu2_conv(x)
        x = self.relu(proj + self.rcu2(proj))

        # RCU 3: 48->64 canales (stride=2)
        proj = self.rcu3_conv(x)
        x = self.relu(proj + self.rcu3(proj))

        # RCU 4: 64->80 canales (stride=2)
        proj = self.rcu4_conv(x)
        x = self.relu(proj + self.rcu4(proj))

        # RCU 5: 80->96 canales (stride=2)
        proj = self.rcu5_conv(x)
        x = self.relu(proj + self.rcu5(proj))

        # Agregación global: pooling promedio + aplanado
        x = self.pool(x)  # AvgPool2d reduce espacial a 1x1
        x = x.view(x.shape[0], -1)  # Flatten para conexión densa
        
        # Rama visual: compresión y procesamiento denso
        x = self.relu(self.dropout(self.fu(x)))   # 384 -> 96 con regularización
        x = self.relu(self.dropout(self.fc0(x)))  # 96 -> 96 con regularización
        
        # Fusión multimodal: concatenar features visuales + ambientales
        x = torch.cat((x, env), dim=1)  # [96] + [1] = [97]
        
        # Capa de salida: regresión lineal sin clamping
        # Permite predicciones fuera de [0, 100] para casos extremos
        output = self.fc_final(x)
        
        return output


