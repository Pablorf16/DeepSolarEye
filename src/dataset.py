"""
dataset.py
==========
Dataset y transformaciones de datos para DeepSolarEye.

Define ``SolarPanelDataset``, un ``torch.utils.data.Dataset`` que carga
imágenes y etiquetas de pérdida de potencia a partir de un fichero CSV
generado por ``data_prep.py``.
"""

import os
import logging

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Configuración del logger (nivel WARNING para no saturar la salida)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SolarPanelDataset(Dataset):
    """
    Dataset de paneles solares para PyTorch.

    Lee un CSV con columnas ``filename`` y ``power_loss`` y carga las
    imágenes desde ``img_dir``. Si una imagen no se encuentra, se
    sustituye por una imagen negra de 224×224 px y se registra una
    advertencia.

    Parameters
    ----------
    csv_path : str
        Ruta al fichero CSV generado por ``data_prep.py``.
    img_dir : str
        Directorio raíz que contiene las imágenes (coincide con RAW_DATA_DIR).
    transform : callable, optional
        Transformación de torchvision aplicada a cada imagen.
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
        """Devuelve el número total de muestras en el dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Devuelve la imagen y la etiqueta correspondientes al índice ``idx``.

        Parameters
        ----------
        idx : int
            Índice de la muestra.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Par (imagen transformada, etiqueta de pérdida de potencia).
        """
        # Obtener la ruta de la imagen a partir del CSV
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        # Cargar la imagen; si falta, usar un placeholder negro para no romper el entrenamiento
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, FileNotFoundError):
            logger.warning(f"Imagen faltante: {img_path}. Usando placeholder negro.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Leer la etiqueta de pérdida de potencia como tensor float32
        label = torch.tensor(
            float(self.data.iloc[idx]['power_loss']),
            dtype=torch.float32,
        )

        # Aplicar las transformaciones de imagen si se han definido
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(phase='train'):
    """
    Devuelve el pipeline de transformaciones estándar para cada fase.

    Durante el entrenamiento se añade un volteo horizontal aleatorio para
    incrementar la variabilidad del dataset (data augmentation).
    La normalización usa las estadísticas de ImageNet.

    Parameters
    ----------
    phase : str
        ``'train'`` para el conjunto de entrenamiento;
        cualquier otro valor para validación/test.

    Returns
    -------
    torchvision.transforms.Compose
        Pipeline de transformaciones listo para usar con el Dataset.
    """
    # Estadísticas de normalización de ImageNet (media y desviación estándar por canal)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),   # Augmentación solo en train
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    # Para validación y test no se aplica augmentación
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
