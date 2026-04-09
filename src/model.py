


import torch
import torch.nn as nn

from src.config import NUM_ENV_FEATURES


class Net(nn.Module):
    """Custom CNN for soiling prediction with environmental feature injection."""

    def __init__(self) -> None:
        """Initialize network architecture."""
        super(Net, self).__init__()

        # Etapa inicial: normalización y feature extraction
        # Conv 7x7 para receptive field grande, BN para estabilidad
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7,
                               padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.AvgPool2d(kernel_size=3)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

        # Unidades residuales: paso de dimensión (1x1) + doble conv 5x5
        # Progresión: 16→32→48→64→80→96 canales, cada una reduce 2x spatial
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

        # Rama visual: 384 features (96 canales × 2×2 spatial) → 96 → 96
        # Dropout evita overfitting en capas densas
        self.fu = nn.Linear(384, 96)
        self.fc0 = nn.Linear(96, 96)
        
        # Salida: concat de rama visual (96) + features ambientales (1)
        # Regresión abierta (sin sigmoid) para diagnóstico irrestricto
        self.fc_final = nn.Linear(96 + NUM_ENV_FEATURES, 1)

    def forward(self, x: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """Forward pass with image and environmental feature processing."""
        # Etapa inicial: normalizar features de entrada
        x = self.relu(self.bn1(self.conv1(x)))

        # Unidades residuales: projection + doble conv + suma residual
        # Cada unidad: dimX → dimX+1 (espacio) y reduce spatial 2×
        proj = self.rcu1_conv(x)
        x = self.relu(proj + self.rcu1(proj))

        proj = self.rcu2_conv(x)
        x = self.relu(proj + self.rcu2(proj))

        proj = self.rcu3_conv(x)
        x = self.relu(proj + self.rcu3(proj))

        proj = self.rcu4_conv(x)
        x = self.relu(proj + self.rcu4(proj))

        proj = self.rcu5_conv(x)
        x = self.relu(proj + self.rcu5(proj))

        # Global pooling: reduce spatial → vector único por muestra
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        
        # Rama visual: 2 capas fully connected con dropout y ReLU
        x = self.relu(self.dropout(self.fu(x)))
        x = self.relu(self.dropout(self.fc0(x)))
        
        # Inyección ambiental: concatenar irradiance directamente
        x = torch.cat((x, env), dim=1)
        
        # Salida: regresión sin activación para rango abierto [−∞, +∞]
        output = self.fc_final(x)
        
        return output


