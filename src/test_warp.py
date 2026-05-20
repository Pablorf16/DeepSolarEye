"""Perspective transformation test for solar panel images.

Applies perspective correction (warp) to correct angled captures.
Output: reports/figures/warp_test/
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
    """Executes perspective transformation test.
    
    Finds image in RAW directory, applies perspective correction
    and saves results in reports/figures/warp_test/.
    
    Returns:
        bool: True if successful, False if error
    """
    logger.info("=" * 50)
    logger.info("PERSPECTIVE TRANSFORMATION TEST")
    logger.info("=" * 50)
    
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "Solar_Panel_Soiling_Image_dataset" / "PanelImages"
    OUTPUT_DIR = BASE_DIR / "reports" / "figures" / "warp_test"
    
    logger.info(f"Searching for images in: {RAW_DATA_DIR}")
    
    # Robust image search (multiple extensions)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [
        path for path in RAW_DATA_DIR.rglob('*')
        if path.is_file() and path.suffix.lower() in valid_extensions
    ]
    
    if not image_files:
        logger.error("No images found in directory.")
        logger.error(f"Supported extensions: {valid_extensions}")
        return False
    
    # Select first image
    image_path = image_files[0]
    logger.info(f"Image found: {image_path.name}")
    
    # Robust reading on Windows (handles special characters)
    img = cv2.imdecode(
        np.fromfile(str(image_path), dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    
    if img is None:
        logger.error("File exists but OpenCV cannot read it.")
        logger.error(f"Verify it is a valid image: {image_path}")
        return False
    
    logger.info("Image loaded successfully. Applying transformation...")
    
    # Geometría: Coordenadas relativas
    h, w = img.shape[:2]
    pts_origen = np.float32([
        [w * 0.15, h * 0.10],   # Top-left
        [w * 0.85, h * 0.15],   # Top-right
        [w * 0.85, h * 0.95],   # Bottom-right
        [w * 0.15, h * 0.90]    # Bottom-left
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
    
    # Draw polygon for visualization
    img_marcada = img.copy()
    cv2.polylines(
        img_marcada,
        [np.int32(pts_origen)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2
    )
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "1_original_marked.jpg"), img_marcada)
    cv2.imwrite(str(OUTPUT_DIR / "2_corrected.jpg"), img_aplanada)
    
    logger.info(f"Transformation completed.")
    logger.info(f"   Original: {image_path.name} ({w}x{h})")
    logger.info(f"   Output:   {OUTPUT_DIR}")
    logger.info(f"   Size:     {IMG_SIZE}x{IMG_SIZE}")
    logger.info("=" * 50)
    
    return True


if __name__ == "__main__":
    probar_recorte()
