"""
test_warp.py - Prueba de transformación de perspectiva para DeepSolarEye v3.0

Aplica corrección de perspectiva (warp) a imágenes de paneles solares.
Útil para pre-procesamiento de imágenes capturadas en ángulo.

Salida: reports/figures/warp_test/
  - 1_marcada.jpg (imagen original con ROI marcado)
  - 2_aplanada.jpg (imagen corregida a 224×224)
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import IMG_SIZE

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def probar_recorte() -> bool:
    """
    Ejecuta prueba de transformación de perspectiva.
    
    Busca una imagen en el directorio RAW, aplica corrección de perspectiva
    y guarda los resultados en reports/figures/warp_test/.
    
    La transformación convierte una región trapezoidal (panel en ángulo)
    en una imagen cuadrada de IMG_SIZE×IMG_SIZE píxeles.
    
    Returns:
        bool: True si éxito, False si error
    
    Example:
        >>> success = probar_recorte()
        >>> print("OK" if success else "Error")
    """
    logger.info("=" * 50)
    logger.info("PRUEBA DE TRANSFORMACIÓN DE PERSPECTIVA (WARP)")
    logger.info("=" * 50)
    
    # Rutas base
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "Solar_Panel_Soiling_Image_dataset" / "PanelImages"
    OUTPUT_DIR = BASE_DIR / "reports" / "figures" / "warp_test"
    
    logger.info(f"Buscando imágenes en: {RAW_DATA_DIR}")
    
    # Búsqueda robusta de imágenes (múltiples extensiones)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [
        path for path in RAW_DATA_DIR.rglob('*')
        if path.is_file() and path.suffix.lower() in valid_extensions
    ]
    
    if not image_files:
        logger.error("No se encontraron imágenes en la carpeta.")
        logger.error(f"Extensiones soportadas: {valid_extensions}")
        return False
    
    # Seleccionar la primera imagen encontrada
    image_path = image_files[0]
    logger.info(f"Imagen encontrada: {image_path.name}")
    
    # Lectura robusta en Windows (maneja rutas con caracteres especiales)
    img = cv2.imdecode(
        np.fromfile(str(image_path), dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    
    if img is None:
        logger.error("El archivo existe, pero OpenCV no puede abrirlo.")
        logger.error(f"Verifica que sea una imagen válida: {image_path}")
        return False
    
    logger.info("Imagen cargada exitosamente. Aplicando transformación...")
    
    # Geometría: coordenadas genéricas relativas a dimensión de imagen
    h, w = img.shape[:2]
    pts_origen = np.float32([
        [w * 0.15, h * 0.10],   # Esquina superior izquierda
        [w * 0.85, h * 0.15],   # Esquina superior derecha
        [w * 0.85, h * 0.95],   # Esquina inferior derecha
        [w * 0.15, h * 0.90]    # Esquina inferior izquierda
    ])
    pts_destino = np.float32([
        [0, 0],
        [IMG_SIZE, 0],
        [IMG_SIZE, IMG_SIZE],
        [0, IMG_SIZE]
    ])
    
    # Transformación de perspectiva
    matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
    img_aplanada = cv2.warpPerspective(img, matriz, (IMG_SIZE, IMG_SIZE))
    
    # Dibujar polígono para visualización
    img_marcada = img.copy()
    cv2.polylines(
        img_marcada,
        [np.int32(pts_origen)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2
    )
    
    # Guardar resultados
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "1_marcada.jpg"), img_marcada)
    cv2.imwrite(str(OUTPUT_DIR / "2_aplanada.jpg"), img_aplanada)
    
    logger.info(f"Transformación completada.")
    logger.info(f"   Original: {image_path.name} ({w}×{h})")
    logger.info(f"   Salida:   {OUTPUT_DIR}")
    logger.info(f"   Tamaño:   {IMG_SIZE}×{IMG_SIZE}")
    logger.info("=" * 50)
    
    return True


if __name__ == "__main__":
    probar_recorte()
