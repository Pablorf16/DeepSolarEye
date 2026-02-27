import os
import torch
import pandas as pd
import logging
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class SolarPanelDataset(Dataset):
    """
    Clase que conecta tus CSVs (creados por data_prep) con PyTorch.
    """
    def __init__(self, csv_path, img_dir, transform=None):
        
        try:
            self.data = pd.read_csv(csv_path)
        except FileNotFoundError:
            logger.error(f"CSV no encontrado: {csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error leyendo CSV {csv_path}: {e}")
            raise
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Busca el nombre de la foto en el CSV
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 2. Abre la imagen
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, FileNotFoundError) as e:
            logger.warning(f"Imagen faltante: {img_path}. Usando placeholder negro.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 3. Lee la etiqueta de potencia (Power Loss)
        # Asumimos que está en la columna 'power_loss'
        label = torch.tensor(float(self.data.iloc[idx]['power_loss']), dtype=torch.float32)

        # 4. Aplica transformaciones
        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(phase='train'):
    """Devuelve las transformaciones estándar para la red neuronal."""
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])