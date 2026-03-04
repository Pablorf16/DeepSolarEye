"""
src/__init__.py - Package initialization para DeepSolarEye v3.0

Hace que el directorio src/ sea un paquete Python importable,
permitiendo:
  1. Imports absolutos: python -m src.train
  2. Ejecución desde cualquier directorio
  3. Importación desde otros packages: from src.model import Net
"""

__version__ = "3.0"
__author__ = "DeepSolarEye Team"

# Importar componentes principales para acceso directo
from src.config import (  # noqa: F401
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    ES_PATIENCE,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
)
from src.model import Net  # noqa: F401
from src.dataset import SolarPanelDataset  # noqa: F401
from src.data_prep import parse_filename_regex, oversample_dataframe  # noqa: F401

__all__ = [
    'SEED',
    'BATCH_SIZE',
    'LEARNING_RATE',
    'MAX_EPOCHS',
    'ES_PATIENCE',
    'SCHEDULER_PATIENCE',
    'SCHEDULER_FACTOR',
    'Net',
    'SolarPanelDataset',
    'parse_filename_regex',
    'oversample_dataframe',
]
