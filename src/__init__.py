"""
src/__init__.py - Package initialization para DeepSolarEye v3.2

Hace que el directorio src/ sea un paquete Python importable,
permitiendo:
  1. Imports absolutos: python -m src.train
  2. Ejecución desde cualquier directorio
  3. Importación explícita: from src.model import Net

NOTA: Este archivo NO importa módulos pesados (torch, pandas, etc.)
para permitir imports ligeros. Use imports explícitos:

    from src.config import BATCH_SIZE     # Solo config (ligero)
    from src.model import Net              # Carga torch
    from src.dataset import SolarPanelDataset  # Carga torch, PIL
"""

__version__ = "3.2"
__author__ = "DeepSolarEye Team"

# Solo metadata, sin imports pesados
# Para usar componentes, importar explícitamente:
#   from src.config import SEED, BATCH_SIZE, ...
#   from src.model import Net
#   from src.dataset import SolarPanelDataset, get_transforms
#   from src.data_prep import parse_filename_regex, oversample_dataframe

__all__ = [
    # Módulos disponibles (importar explícitamente)
    'config',
    'model',
    'dataset',
    'data_prep',
    'train',
    'eda',
    'plot_results',
    'test_warp',
]
