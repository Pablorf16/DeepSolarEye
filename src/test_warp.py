# src/test_warp.py
import cv2
import numpy as np
import os

def probar_recorte():
    print("--- INICIANDO PRUEBA DE RECORTE (ROI) ---")
    
    # 1. Rutas base
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_NAME = "solar_Fri_Jun_16_6__0__25_2017_L_0.0901960784314_I_0.003.jpg"
    
    # 2. LA RUTA CORREGIDA: Añadimos las subcarpetas que encontraste
    IMAGE_PATH = os.path.join(
        BASE_DIR, 
        "data", 
        "raw", 
        "Solar_Panel_Soiling_Image_dataset", 
        "PanelImages", 
        IMAGE_NAME
    )
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "figures", "warp_test")
    
    print(f"Buscando imagen en:\n{IMAGE_PATH}")
    
    # 3. Comprobación de existencia
    if not os.path.exists(IMAGE_PATH):
        print("\n[ERROR CRÍTICO] Sigo sin encontrar el archivo.")
        print("Revisa si el nombre de la imagen termina en .jpg o si Windows ocultó la extensión.")
        return

    # 4. Lectura a prueba de fallos en Windows
    img = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("\n[ERROR] El archivo existe, pero OpenCV no puede abrirlo.")
        return

    print("¡Imagen encontrada y cargada! Aplicando Transformación de Perspectiva...")

    # 5. Geometría (Tus coordenadas)
    pts_origen = np.float32([[80, 19], [190, 35], [153, 162], [16, 140]])
    pts_destino = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])
    
    # 6. Transformación matemática
    matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
    img_aplanada = cv2.warpPerspective(img, matriz, (224, 224))
    
    # 7. Dibujar el polígono para visualización
    img_marcada = img.copy()
    cv2.polylines(img_marcada, [np.int32(pts_origen)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # 8. Guardar
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_marcada.jpg"), img_marcada)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_aplanada.jpg"), img_aplanada)
    
    print(f"\n✅ ¡Transformación completada con éxito!")
    print(f"Revisa las imágenes en: {OUTPUT_DIR}")

if __name__ == "__main__":
    probar_recorte()