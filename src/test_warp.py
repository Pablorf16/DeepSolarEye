# src/test_warp.py

import cv2
import numpy as np
import os
from pathlib import Path

def probar_recorte():
    print("--- INICIANDO PRUEBA DE RECORTE (ROI) ---")
    
    # 1. Rutas base
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "Solar_Panel_Soiling_Image_dataset" / "PanelImages"
    
    print(f"Buscando imágenes en:\n{RAW_DATA_DIR}")
    
    # 2. Búsqueda ROBUSTA de imágenes (múltiples extensiones)
    # Soporta: .jpg, .jpeg, .png, .JPG, .JPEG, .PNG (case-insensitive)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for path in RAW_DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            image_files.append(path)
    
    if not image_files:
        print("\n[ERROR] No se encontraron imágenes en la carpeta.")
        print(f"Extensiones soportadas: .jpg, .jpeg, .png")
        print(f"Verifica que existan imágenes en:\n{RAW_DATA_DIR}")
        return
    
    # Seleccionar la primera imagen encontrada
    IMAGE_PATH = image_files[0]
    print(f"✅ Imagen encontrada: {IMAGE_PATH.name}\n")
    
    OUTPUT_DIR = BASE_DIR / "reports" / "figures" / "warp_test"
    
    # 3. Lectura robusta en Windows
    img = cv2.imdecode(np.fromfile(str(IMAGE_PATH), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("\n[ERROR] El archivo existe, pero OpenCV no puede abrirlo.")
        print(f"Verifica que sea una imagen válida: {IMAGE_PATH}")
        return

    print("✅ Imagen cargada exitosamente. Aplicando Transformación de Perspectiva...")
    
    # 4. Geometría (Coordenadas genéricas aproximadas, relativas a dimensión de imagen)
    h, w = img.shape[:2]
    pts_origen = np.float32([
        [w * 0.15, h * 0.10],   # Esquina superior izquierda
        [w * 0.85, h * 0.15],   # Esquina superior derecha
        [w * 0.85, h * 0.95],   # Esquina inferior derecha
        [w * 0.15, h * 0.90]    # Esquina inferior izquierda
    ])
    pts_destino = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])
    
    # 5. Transformación matemática
    matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
    img_aplanada = cv2.warpPerspective(img, matriz, (224, 224))
    
    # 6. Dibujar el polígono para visualización
    img_marcada = img.copy()
    cv2.polylines(img_marcada, [np.int32(pts_origen)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # 7. Guardar resultados
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    cv2.imwrite(str(OUTPUT_DIR / "1_marcada.jpg"), img_marcada)
    cv2.imwrite(str(OUTPUT_DIR / "2_aplanada.jpg"), img_aplanada)
    
    print(f"\n✅ ¡Transformación completada con éxito!")
    print(f"   Original detectado: {IMAGE_PATH.name}")
    print(f"   Dimensiones: {w}×{h}")
    print(f"   Salida: {OUTPUT_DIR}")

if __name__ == "__main__":
    probar_recorte()