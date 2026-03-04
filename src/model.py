"""
model.py
========
Definición de la arquitectura CNN para DeepSolarEye.

La clase ``Net`` es una adaptación de ImpactNet para regresión de la pérdida
de potencia (0-100 %) a partir de imágenes RGB de paneles solares.

Arquitectura:
    - Capa inicial: Conv2d(3→16, 7×7)
    - 5 Analysis Units (AU1-AU5) con conexiones residuales y BatchNorm
    - Capas fully-connected: 384→96→96→1
    - Salida: sigmoid × 100 → rango [0, 100] %
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Red neuronal convolucional adaptada de ImpactNet para DeepSolarEye.

    Entrada : tensor RGB de forma (N, 3, 224, 224).
    Salida  : tensor de forma (N, 1) con la pérdida de potencia en %.
    """

    def __init__(self):
        super(Net, self).__init__()

        # ------------------------------------------------------------------
        # Bloque de extracción de características
        # ------------------------------------------------------------------

        # Capa inicial: extrae características de bajo nivel con kernel 7×7
        self.conv1 = nn.Conv2d(3, 16, 7, padding=3)
        self.pool = nn.AvgPool2d(3)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        # AU1 (Analysis Unit 1) — 16→32 canales
        self.rcu1_conv = nn.Conv2d(16, 32, 1, 2)
        self.rcu1 = nn.Sequential(
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32),
        )

        # AU2 — 32→48 canales
        self.rcu2_conv = nn.Conv2d(32, 48, 1, 2)
        self.rcu2 = nn.Sequential(
            nn.Conv2d(48, 48, 5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, 5, padding=2),
            nn.BatchNorm2d(48),
        )

        # AU3 — 48→64 canales
        self.rcu3_conv = nn.Conv2d(48, 64, 1, 2)
        self.rcu3 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
        )

        # AU4 — 64→80 canales
        self.rcu4_conv = nn.Conv2d(64, 80, 1, 2)
        self.rcu4 = nn.Sequential(
            nn.Conv2d(80, 80, 5, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 80, 5, padding=2),
            nn.BatchNorm2d(80),
        )

        # AU5 — 80→96 canales
        self.rcu5_conv = nn.Conv2d(80, 96, 1, 2)
        self.rcu5 = nn.Sequential(
            nn.Conv2d(96, 96, 5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 5, padding=2),
            nn.BatchNorm2d(96),
        )

        # ------------------------------------------------------------------
        # Capas totalmente conectadas (clasificador / regresor)
        # ------------------------------------------------------------------
        # Tamaño del tensor aplanado tras AvgPool2d(3):
        #   rcu5_conv produce 7×7×96  →  AvgPool2d(3) → 2×2×96 = 384
        self.fu = nn.Linear(384, 96)
        self.fc0 = nn.Linear(96, 96)

        # Capa de regresión final: produce un único valor de pérdida de potencia
        self.fc_final = nn.Linear(96, 1)

    def forward(self, x):
        """
        Paso hacia adelante de la red.

        Parameters
        ----------
        x : torch.Tensor
            Lote de imágenes de forma (N, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Predicciones de pérdida de potencia (%) de forma (N, 1),
            acotadas en el rango [0, 100].
        """
        # 1. Extracción inicial de características
        x = self.conv1(x)

        # 2. Bloques residuales: cada AU suma la proyección de entrada con
        #    las características aprendidas por el bloque convolucional
        x = self.relu(self.rcu1_conv(x) + self.rcu1(self.rcu1_conv(x)))
        x = self.relu(self.rcu2_conv(x) + self.rcu2(self.rcu2_conv(x)))
        x = self.relu(self.rcu3_conv(x) + self.rcu3(self.rcu3_conv(x)))
        x = self.relu(self.rcu4_conv(x) + self.rcu4(self.rcu4_conv(x)))
        x = self.relu(self.rcu5_conv(x) + self.rcu5(self.rcu5_conv(x)))

        # 3. Pooling global y aplanado a vector 1-D
        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        # 4. Clasificador fully-connected con dropout para regularización
        x = self.relu(self.dropout(self.fu(x)))
        x = self.relu(self.dropout(self.fc0(x)))

        # 5. Regresión final: sigmoid escala la salida al rango [0, 100] %
        output = torch.sigmoid(self.fc_final(x)) * 100.0
        return output
