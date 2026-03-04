"""
dataset.py - Cargador de datos para DeepSolarEye v3.0

PyTorch Dataset para cargar imágenes de paneles solares desde archivos CSV
con transformaciones y validación robusta de rutas.
"""

import logging
import os
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Importar configuración centralizada
from src.config import (
    AUGMENTATION_STRATEGY,
    CATEGORY_BINS,
    CATEGORY_LABELS,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SolarPanelDataset(Dataset):
    """
    PyTorch Dataset agnóstico a balanceo de clases para DeepSolarEye v3.0.
    
    Carga imágenes de paneles solares desde CSV y aplica transformaciones
    según la fase (entrenamiento, validación o test).
    
    ARQUITECTURA v3.0 - Separación de responsabilidades:
    - Oversampling: Realizado en data_prep.py → train_dataset.csv contiene
      filas duplicadas estratégicamente
    - Dataset: Completamente agnóstico a si está oversampleado o no
    - Augmentation: get_transforms() aplica durante training
    - Val/Test: CSVs sin oversample, transformaciones determinísticas
    
    VENTAJA: Un único dataset para train/val/test, cumple principio DRY.
    El balanceo está en los DATOS (CSV), no en lógica de muestreo.
    
    MANEJO DE RUTAS:
    - img_dir: Ruta base absoluta o relativa
    - CSV contiene: rutas relativas desde img_dir
      Ej: Solar_Panel_Soiling_Image_dataset/PanelImages/panel_001.jpg
    - Ruta final: img_dir / filename
    
    VALIDACIÓN: Comprueba que img_dir existe; si falla, lanza error explícito.
    
    Attributes:
        data (pd.DataFrame): DataFrame con columnas [filename, power_loss, ...]
        img_dir (str): Ruta base del directorio de imágenes
        transform (Callable): Transformaciones de torchvision a aplicar
        category (pd.Series): Categorías discretizadas (Limpio/Leve/...)
    """
    
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[Callable] = None,
        verbose: bool = False,
    ) -> None:
        """
        Inicialización del dataset.
        
        Args:
            csv_path (str): Ruta al archivo CSV con metadatos.
                           Columnas requeridas: [filename, power_loss]
                           Example: 'data/processed/train_dataset.csv'
            
            img_dir (str): Ruta base del directorio de imágenes.
                          Puede ser absoluta o relativa.
                          Example: 'data/raw'
            
            transform (Callable, optional): Función de transformación
                                          (ej: get_transforms('train')).
                                          Si None, retorna PIL Image. (default: None)
            
            verbose (bool): Si True, imprime distribución de categorías
                           al cargar (útil para debugging). (default: False)
        
        Raises:
            FileNotFoundError: Si csv_path no existe.
            ValueError: Si img_dir no es un directorio válido.
            KeyError: Si CSV no contiene columna 'filename' o 'power_loss'.
        
        Example:
            >>> dataset = SolarPanelDataset(
            ...     csv_path='data/processed/train_dataset.csv',
            ...     img_dir='data/raw',
            ...     transform=get_transforms('train'),
            ...     verbose=True
            ... )
            >>> print(len(dataset))  # número total de muestras
        """
        
        # 1. Cargar CSV con validación
        try:
            self.data = pd.read_csv(csv_path)
        except FileNotFoundError:
            logger.error(f"CSV no encontrado: {csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error leyendo CSV {csv_path}: {e}")
            raise
        
        # Verificar columnas requeridas
        required_cols = ['filename', 'power_loss']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise KeyError(
                f"Columnas faltantes en CSV: {missing_cols}\n"
                f"Columnas encontradas: {list(self.data.columns)}"
            )
        
        # 2. Validar y expandir ruta del directorio
        img_dir_path = os.path.expanduser(os.path.abspath(img_dir))
        if not os.path.isdir(img_dir_path):
            raise ValueError(
                f"Directorio de imágenes no válido:\n"
                f"  Especificado: {img_dir}\n"
                f"  Expandido: {img_dir_path}\n"
                f"Asegúrate que existe: data/raw/Solar_Panel_Soiling_Image_dataset/PanelImages/"
            )
        
        self.img_dir = img_dir_path
        self.transform = transform
        
        # 3. Crear columna de categorías (para reporting y análisis)
        # Discretizar power_loss (continuo) en categorías (discreto)
        self.data['category'] = pd.cut(
            self.data['power_loss'],
            bins=CATEGORY_BINS,
            labels=CATEGORY_LABELS,
            include_lowest=True,
        )
        
        # 4. Opcionalmente, reportar distribución de categorías
        if verbose:
            self._report_distribution()
    
    def _report_distribution(self) -> None:
        """
        Imprime distribución de categorías en el CSV cargado.
        
        Útil para verificar que oversampling se aplicó correctamente
        o para entender el balance del dataset.
        """
        print(f"\n📊 DISTRIBUCIÓN DEL CSV CARGADO: {len(self.data)} ejemplos")
        for label in CATEGORY_LABELS:
            count = (self.data['category'] == label).sum()
            pct = 100 * count / len(self.data) if len(self.data) > 0 else 0
            # Barra visual (cada 2% = un bloque █)
            bar = "█" * int(pct / 2)
            print(f"   {label:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    def __len__(self) -> int:
        """Retorna el número total de muestras en el dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Obtiene una muestra (imagen, etiqueta) por índice.
        
        Args:
            idx (int): Índice de la muestra (0 <= idx < len(self))
        
        Returns:
            tuple: (image, label) donde:
                - image (torch.Tensor): Imagen procesada [3, 224, 224]
                  Valores normalizados con ImageNet mean/std
                - label (torch.Tensor): Pérdida de potencia [0-100]
                  dtype: torch.float32
        
        Notes:
            Si una imagen no existe o está corrupta, retorna una imagen
            negra [0, 0, 0] para no romper el entrenamiento. Se loguea
            una advertencia.
        
        Raises:
            IndexError: Si idx está fuera de rango.
        
        Example:
            >>> dataset = SolarPanelDataset(...)
            >>> image, label = dataset[0]
            >>> print(image.shape, label.item())
            torch.Size([3, 224, 224]) 12.5
        """
        
        # 1. Obtener nombre de archivo del CSV (ruta relativa)
        relative_filename = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, relative_filename)
        
        # 2. Abrir imagen con manejo robusto de errores
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, FileNotFoundError) as e:
            # Log de advertencia: imagen no encontrada
            logger.warning(
                f"Imagen no encontrada o corrupta: {img_path}. "
                f"Usando placeholder negro para no interrumpir entrenamiento."
            )
            # Crear imagen negra de respaldo (sustituto seguro)
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        # 3. Leer etiqueta (Power Loss en %)
        label = torch.tensor(
            float(self.data.iloc[idx]['power_loss']),
            dtype=torch.float32
        )

        # 4. Aplicar transformaciones (augmentation si train, norm si test)
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(phase: str = 'train') -> transforms.Compose:
    """
    Define transformaciones y augmentación de datos según la fase.
    
    ARQUITECTURA:
    - Training: Augmentación agresiva para aprender características robustas
    - Validation/Test: Solo normalización (determinístico, reproducible)
    
    AUGMENTACIÓN EN v3.0:
    - RandomHorizontalFlip(p=0.5): Volteo horizontal (panel funciona igual)
    - RandomVerticalFlip(p=0.5): Volteo vertical (panel funciona igual)
    - RandomRotation(degrees=180): Rotación CONTINUA [-180°, +180°]
    
    JUSTIFICACIÓN DE ROTACIÓN CONTINUA (vs discreta):
    ────────────────────────────────────────────────
    Con oversampling, tenemos múltiples copias de la misma imagen.
    
    Rotación DISCRETA [0°, 90°, 180°, 270°]:
    - Solo 4 variaciones únicas
    - Copias oversampleadas generan solo 4 imágenes distintas
    - Memorización: modelo aprende 4 configuraciones exactas
    
    Rotación CONTINUA [0°, 360°]:
    - Infinitas variaciones (12.3°, 45.7°, 179.9°, etc.)
    - Cada copia oversampleada tiene ángulo aleatorio único
    - Prevención de memorización: modelo debe aprender features reales
    
    FÍSICAMENTE: Un panel sucio mantiene patrones de suciedad en
    cualquier ángulo. Por lo tanto, rotación física es válida.
    
    Args:
        phase (str): 'train' (con augmentación) o 'test'/'val'
                    (sin augmentación, solo normalización).
                    (default: 'train')
    
    Returns:
        transforms.Compose: Pipeline de transformaciones encadenadas.
    
    Raises:
        ValueError: Si phase no es 'train', 'val' o 'test'.
    
    Example:
        >>> train_transform = get_transforms('train')
        >>> test_transform = get_transforms('test')
        >>> 
        >>> from PIL import Image
        >>> img = Image.open('sample.jpg')
        >>> img_tensor = train_transform(img)  # [3, 224, 224]
    """
    
    if phase == 'train':
        # Augmentación agresiva para entrenamiento
        return transforms.Compose([
            # Redimensiona a 224×224 (tamaño entrada modelo)
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # Volteo horizontal aleatorio (p=50%)
            transforms.RandomHorizontalFlip(p=AUGMENTATION_STRATEGY['horizontal_flip']),
            # Volteo vertical aleatorio (p=50%)
            transforms.RandomVerticalFlip(p=AUGMENTATION_STRATEGY['vertical_flip']),
            # Rotación CONTINUA en rango [-180°, +180°] = círculo completo
            # torchvision.transforms.RandomRotation(degrees=AUGMENTATION_STRATEGY['rotation_degrees']) rota dentro
            # de este rango de forma aleatoria uniforme
            transforms.RandomRotation(degrees=AUGMENTATION_STRATEGY['rotation_degrees']),
            # Convierte PIL Image → torch.Tensor en rango [0, 1]
            transforms.ToTensor(),
            # Normaliza con parámetros ImageNet (media y desv. estándar)
            # Valores: imagen normalizada = (imagen - mean) / std
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Sin augmentación para validación/test (evaluación determinística)
        return transforms.Compose([
            # Solo redimensionar
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # Convertir a tensor
            transforms.ToTensor(),
            # Normalizar
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])



