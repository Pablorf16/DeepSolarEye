"""
eda.py - Análisis Exploratorio de Datos para DeepSolarEye v3.0

Genera visualizaciones del dataset para análisis y documentación.
Utiliza configuración centralizada de config.py.

Salida: reports/figures/
  - power_loss_histogram.png
  - power_loss_by_category.png
  - irradiance_histogram.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CATEGORY_BINS, CATEGORY_LABELS

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas dinámicas
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'reports' / 'figures'


def plot_power_loss_distribution(csv_path: Path, output_dir: Path) -> bool:
    """
    Genera histograma de distribución de pérdida de potencia.
    
    Permite detectar desbalanceo de clases y sesgos en el dataset.
    Útil para justificar estrategia de oversampling en memoria TFG.
    
    Args:
        csv_path (Path): Ruta al CSV con columna 'power_loss'
        output_dir (Path): Directorio donde guardar la figura
    
    Returns:
        bool: True si éxito, False si error
    
    Example:
        >>> plot_power_loss_distribution(
        ...     Path('data/processed/train_dataset.csv'),
        ...     Path('reports/figures/')
        ... )
        True
    """
    try:
        df = pd.read_csv(csv_path)
        
        if 'power_loss' not in df.columns:
            logger.error(f"Columna 'power_loss' no encontrada en {csv_path}")
            return False
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 6))
        
        # Histograma con KDE
        sns.histplot(df['power_loss'], bins=50, kde=True, color='royalblue')
        
        # Líneas verticales para límites de categorías
        for i, limit in enumerate(CATEGORY_BINS[1:-1]):
            plt.axvline(x=limit, color='red', linestyle='--', alpha=0.7)
            plt.text(limit + 1, plt.ylim()[1] * 0.9, CATEGORY_LABELS[i], 
                     fontsize=9, color='darkred')
        
        plt.title('Distribución de Pérdida de Potencia - DeepSolarEye v3.0', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Pérdida de Potencia (%)', fontsize=12)
        plt.ylabel('Frecuencia (Número de Imágenes)', fontsize=12)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'power_loss_histogram.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Histograma guardado en: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generando histograma: {e}")
        return False


def plot_category_distribution(csv_path: Path, output_dir: Path) -> bool:
    """
    Genera gráfico de barras por categoría de suciedad.
    
    Visualiza balance entre categorías (Limpio/Leve/Moderado/Alto/Crítico).
    Justifica necesidad de oversampling si hay desbalance.
    
    Args:
        csv_path (Path): Ruta al CSV con columna 'dirt_category'
        output_dir (Path): Directorio donde guardar la figura
    
    Returns:
        bool: True si éxito, False si error
    """
    try:
        df = pd.read_csv(csv_path)
        
        if 'dirt_category' not in df.columns:
            logger.warning("Columna 'dirt_category' no encontrada, creándola...")
            df['dirt_category'] = pd.cut(
                df['power_loss'],
                bins=CATEGORY_BINS,
                labels=CATEGORY_LABELS,
                include_lowest=True
            )
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Contar por categoría y ordenar
        category_counts = df['dirt_category'].value_counts().reindex(CATEGORY_LABELS).fillna(0).astype(int)
        
        # Colores por severidad
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
        
        bars = plt.bar(CATEGORY_LABELS, category_counts, color=colors, edgecolor='black')
        
        # Etiquetas en barras
        for bar, count in zip(bars, category_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{int(count)}', ha='center', fontsize=10, fontweight='bold')
        
        plt.title('Distribución por Categoría de Suciedad - DeepSolarEye v3.0',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Categoría de Suciedad', fontsize=12)
        plt.ylabel('Número de Imágenes', fontsize=12)
        plt.xticks(rotation=0, fontsize=11)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'power_loss_by_category.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de categorías guardado en: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generando gráfico de categorías: {e}")
        return False


def plot_irradiance_distribution(csv_path: Path, output_dir: Path) -> bool:
    """
    Genera histograma de irradiancia solar (si disponible).
    
    Args:
        csv_path (Path): Ruta al CSV con columna 'irradiance' (opcional)
        output_dir (Path): Directorio donde guardar la figura
    
    Returns:
        bool: True si éxito o columna no disponible, False si error
    """
    try:
        df = pd.read_csv(csv_path)
        
        if 'irradiance' not in df.columns:
            logger.warning("Columna 'irradiance' no encontrada. Saltando.")
            return True
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        sns.histplot(df['irradiance'], bins=30, kde=True, color='darkorange')
        
        plt.title('Distribución de Irradiancia Solar - DeepSolarEye v3.0',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Irradiancia (W/m²)', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'irradiance_histogram.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Histograma de irradiancia guardado en: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generando histograma de irradiancia: {e}")
        return False


def run_eda() -> None:
    """
    Ejecuta análisis exploratorio completo del dataset.
    
    Genera todas las visualizaciones en reports/figures/.
    Requiere haber ejecutado data_prep.py primero.
    """
    logger.info("=" * 60)
    logger.info("ANÁLISIS EXPLORATORIO DE DATOS (EDA) - DeepSolarEye v3.0")
    logger.info("=" * 60)
    
    train_csv = PROCESSED_DIR / 'train_dataset.csv'
    
    if not train_csv.exists():
        logger.error(f"No se encuentra: {train_csv}")
        logger.error("Ejecuta primero: python src/data_prep.py")
        return
    
    logger.info(f"Dataset: {train_csv}")
    logger.info(f"Output:  {OUTPUT_DIR}")
    
    # Generar visualizaciones
    plot_power_loss_distribution(train_csv, OUTPUT_DIR)
    plot_category_distribution(train_csv, OUTPUT_DIR)
    plot_irradiance_distribution(train_csv, OUTPUT_DIR)
    
    logger.info("=" * 60)
    logger.info("EDA COMPLETADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_eda()

