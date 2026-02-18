import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Versión adaptada de ImpactNet para DeepSolarEye.
    - Entrada: Solo Imagen RGB (se eliminó el factor ambiental temporalmente).
    - Salida: 1 valor (Regresión de Power Loss).
    """
    def __init__(self):
        super(Net, self).__init__()
        
        # --- BLOQUE DE EXTRACCIÓN DE CARACTERÍSTICAS (Igual al original) ---
        # Capa inicial
        self.conv1 = nn.Conv2d(3, 16, 7, padding=3)
        self.pool = nn.AvgPool2d(3)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        # AU1 (Analysis Unit 1)
        self.rcu1_conv = nn.Conv2d(16, 32, 1, 2)
        self.rcu1 = nn.Sequential(
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32)
        )
        
        # AU2
        self.rcu2_conv = nn.Conv2d(32, 48, 1, 2)
        self.rcu2 = nn.Sequential(
            nn.Conv2d(48, 48, 5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, 5, padding=2),
            nn.BatchNorm2d(48)
        )
        
        # AU3
        self.rcu3_conv = nn.Conv2d(48, 64, 1, 2)
        self.rcu3 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64)
        )

        # AU4
        self.rcu4_conv = nn.Conv2d(64, 80, 1, 2)
        self.rcu4 = nn.Sequential(
            nn.Conv2d(80, 80, 5, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.Conv2d(80, 80, 5, padding=2),
            nn.BatchNorm2d(80)
        )
        
        # AU5
        self.rcu5_conv = nn.Conv2d(80, 96, 1, 2)
        self.rcu5 = nn.Sequential(
            nn.Conv2d(96, 96, 5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 5, padding=2),
            nn.BatchNorm2d(96)
        )

        # --- CAPAS TOTALMENTE CONECTADAS (Adaptadas) ---
        # El original tenía 384 entradas tras aplanar, reducimos a 96
        self.fu = nn.Linear(384, 96)
        self.fc0 = nn.Linear(96, 96)
        
        # Capa Final de Regresión
        # Originalmente combinaba con 'ef' (env factors). 
        # Aquí simplificamos para salir directo a 1 valor.
        self.fc_final = nn.Linear(96, 1) 

    def forward(self, x):
        # x es la imagen [Batch, 3, 224, 224]
        
        # 1. Capa inicial
        x = self.conv1(x)
        
        # 2. Bloques Residuales (La magia de DeepSolarEye)
        # Cada bloque suma la entrada convolucionada con el residuo
        x = self.relu(self.rcu1_conv(x) + self.rcu1(self.rcu1_conv(x)))
        x = self.relu(self.rcu2_conv(x) + self.rcu2(self.rcu2_conv(x)))
        x = self.relu(self.rcu3_conv(x) + self.rcu3(self.rcu3_conv(x)))
        x = self.relu(self.rcu4_conv(x) + self.rcu4(self.rcu4_conv(x)))
        x = self.relu(self.rcu5_conv(x) + self.rcu5(self.rcu5_conv(x)))
        
        # 3. Pooling y Aplanado
        x = self.pool(x)
        x = x.view(x.shape[0], -1) # Flatten
        
        # 4. Clasificador (Fully Connected)
        x = self.relu(self.dropout(self.fu(x)))
        x = self.relu(self.dropout(self.fc0(x)))
        
        # 5. Salida Final
        x = self.fc_final(x)
        
        return x