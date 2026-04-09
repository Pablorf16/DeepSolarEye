

import logging
import os
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import (
    AUGMENTATION_STRATEGY,
    CATEGORY_BINS,
    CATEGORY_LABELS,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SolarPanelDataset(Dataset):
    
    
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[Callable] = None,
        verbose: bool = False,
    ) -> None:
        
        try:
            self.data = pd.read_csv(csv_path)
        except FileNotFoundError:
            logger.error(f"CSV no encontrado: {csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error al leer CSV {csv_path}: {e}")
            raise

        required_cols = ['filename', 'power_loss', 'irradiance']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise KeyError(f"Columnas faltantes: {missing_cols}")

        img_dir_path = os.path.expanduser(os.path.abspath(img_dir))
        if not os.path.isdir(img_dir_path):
            raise ValueError(f"Directorio inválido: {img_dir}")
        
        self.img_dir = img_dir_path
        self.transform = transform
        
        self.data['category'] = pd.cut(
            self.data['power_loss'],
            bins=CATEGORY_BINS,
            labels=CATEGORY_LABELS,
            include_lowest=True,
        )
        
        if verbose:
            self._report_distribution()
    
    def _report_distribution(self) -> None:
        """Imprime distribución de categorías para verificación."""
        print(f"\nDistribución del dataset ({len(self.data)} muestras):")
        for label in CATEGORY_LABELS:
            count = (self.data['category'] == label).sum()
            pct = 100 * count / len(self.data) if len(self.data) > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    def __len__(self) -> int:
        """Retorna el total de muestras del dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Obtiene muestra (imagen, etiqueta, features ambientales) por índice.
        
        Retorna: (image, label, env_features) donde:
            - image [3, 224, 224]: RGB normalizado con ImageNet
            - label: pérdida de potencia [0-100]%
            - env_features: irradiance normalizada [0-1]
        
        Si la imagen no existe o está corrupta, retorna imagen negra
        como respaldo sin interrumpir el entrenamiento.
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        # Abrir imagen con manejo de errores
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, FileNotFoundError):
            logger.warning(f"Imagen no encontrada: {img_path}. Usando placeholder.")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        # Leer etiqueta y features ambientales
        label = torch.tensor(float(row['power_loss']), dtype=torch.float32)
        env_features = torch.tensor(
            [float(row['irradiance'])],
            dtype=torch.float32,
        )

        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)

        return image, label, env_features


def get_transforms(phase: str = 'train') -> transforms.Compose:
    """Define transformaciones de datos según fase.
    
    Args:
        phase: 'train' (con aumentación) o 'test'/'val' (solo normalización)
    
    Returns:
        transforms.Compose: Pipeline de transformaciones encadenadas
    
    Raises:
        ValueError: Si phase no es 'train', 'val' o 'test'
    """
    # Validar fase
    valid_phases = {'train', 'val', 'test'}
    if phase not in valid_phases:
        raise ValueError(f"phase debe ser 'train', 'val' o 'test', recibido: '{phase}'")
    
    # Transformaciones: redimensionar, convertir a tensor, normalizar
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])




