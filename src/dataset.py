"""Definición del Dataset para el pipepline de DeepSolarEye v4.0

Gestiona la carga de imágenes, la integración de datos ambientales y la categorización dinámica basada en cuartiles de pérdida de potencia"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import (
    CATEGORY_BINS,
    CATEGORY_LABELS,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Ruta al archivo de cuartiles dinámicos (generado por data_prep.py)
BASE_DIR = Path(__file__).resolve().parent.parent
QUARTILES_FILE = BASE_DIR / "data" / "quartiles.json"


def _load_quartiles() -> tuple:
    """Carga cuartiles dinámicos desde data/quartiles.json.
    
    Returns:
        tuple: (bins, labels) del dataset actual
               Si no existe, retorna bins/labels fijos de config.py
    """
    if QUARTILES_FILE.exists():
        try:
            with open(QUARTILES_FILE, 'r', encoding='utf-8') as f:
                q_data = json.load(f)
            bins = q_data.get('bins', CATEGORY_BINS)
            labels = q_data.get('labels', CATEGORY_LABELS)
            logger.info("Dynamic quartiles loaded from %s", QUARTILES_FILE)
            return bins, labels
        except Exception as e:
            logger.warning("Error loading dynamic quartiles: %s. Using fixed bins from config.py", e)
            return CATEGORY_BINS, CATEGORY_LABELS
    else:
        logger.warning("Quartiles file not found: %s. Using fixed bins from config.py", QUARTILES_FILE)
        return CATEGORY_BINS, CATEGORY_LABELS


class SolarPanelDataset(Dataset):
    """Dataset personalizado para imágenes de paneles solares con features ambientales.
    
    Carga imágenes, irradiancia y categoría de suciedad. Soporta categorización dinámica
    mediante cuartiles o categoría precomputada en el CSV.
    """


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
            logger.error("CSV not found: %s", csv_path)
            raise
        except Exception as e:
            logger.error("Error reading CSV %s: %s", csv_path, e)
            raise

        # Validar columnas requeridas
        required_cols = ['filename', 'power_loss', 'irradiance']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise KeyError(f"Missing columns: {missing_cols}")

        # Validar directorio de imágenes
        img_dir_path = Path(img_dir).expanduser().resolve()
        if not img_dir_path.is_dir():
            raise ValueError(f"Invalid directory: {img_dir}")
        
        self.img_dir = str(img_dir_path)
        self.transform = transform
        
        # Estratificación: usar categoría existente (v4.0) o recalcular con cuartiles dinámicos
        if 'dirt_category' in self.data.columns:
            logger.info("Column 'dirt_category' found in CSV. Using existing categorization.")
            self.category_labels = self.data['dirt_category'].unique()
        else:
            # Si no existe, recalcular con cuartiles dinámicos
            logger.info("Column 'dirt_category' not found. Recalculating with dynamic quartiles.")
            bins, labels = _load_quartiles()
            self.category_labels = labels
            
            self.data['category'] = pd.cut(
                self.data['power_loss'],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )
        
        if verbose:
            self._report_distribution()
    
    def _report_distribution(self) -> None:
        """Imprime distribución de categorías para verificación."""
        print(f"\nDataset distribution ({len(self.data)} samples):")
        
        # Determinar columna de categoría (v4.0 usa 'dirt_category', sino 'category')
        cat_col = 'dirt_category' if 'dirt_category' in self.data.columns else 'category'
        
        for label in self.category_labels:
            count = (self.data[cat_col] == label).sum()
            pct = 100 * count / len(self.data) if len(self.data) > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    def __len__(self) -> int:
        """Retorna el total de muestras del dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Retorna tupla (imagen, etiqueta_pérdida, features_ambientales) para índice dado."""

        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        # Cargar imagen con manejo de excepciones
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, FileNotFoundError):
            logger.warning("Image not found: %s. Using black placeholder.", img_path)
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        # Convertir etiqueta (pérdida %) y features ambientales a tensores
        label = torch.tensor(float(row['power_loss']), dtype=torch.float32)
        env_features = torch.tensor(
            [float(row['irradiance'])],
            dtype=torch.float32,
        )

        # Aplicar transformaciones determinísticas
        if self.transform:
            image = self.transform(image)

        return image, label, env_features


def get_transforms(phase: str = 'train') -> transforms.Compose:
    
    # Validar fase
    valid_phases = {'train', 'val', 'test'}
    if phase not in valid_phases:
        raise ValueError(f"phase must be 'train', 'val' or 'test', got: '{phase}'")
    
    # Pipeline determinístico idéntico para todas las fases
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])




