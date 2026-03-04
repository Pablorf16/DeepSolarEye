"""
test_warp.py
============
Prueba de la transformación de perspectiva (warp) sobre una imagen de panel.

Aplica una homografía para "aplanar" la perspectiva de una imagen de panel
solar y guarda el resultado en ``reports/figures/warp_test/``.

Uso::

    python src/test_warp.py
"""

import os

import cv2
import numpy as np


def test_perspective_warp():
    """
    Carga una imagen de panel, aplica una transformación de perspectiva
    y guarda las imágenes marcada y aplanada en el directorio de salida.

    La función imprime mensajes de progreso y finaliza sin lanzar excepciones
    si la imagen de entrada no se encuentra o no puede abrirse.
    """
    print("--- INICIANDO PRUEBA DE RECORTE (ROI) ---")

    # Rutas del proyecto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_NAME = "solar_Fri_Jun_16_6__0__25_2017_L_0.0901960784314_I_0.003.jpg"
    IMAGE_PATH = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "Solar_Panel_Soiling_Image_dataset",
        "PanelImages",
        IMAGE_NAME,
    )
    OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "figures", "warp_test")

    print(f"Buscando imagen en:\n{IMAGE_PATH}")

    if not os.path.exists(IMAGE_PATH):
        print("\n[ERROR] No se encuentra el archivo de imagen.")
        print("Verifica la ruta y la extensión del fichero.")
        return

    # Leer la imagen usando fromfile para compatibilidad con rutas Unicode
    img = cv2.imdecode(np.fromfile(IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("\n[ERROR] El archivo existe, pero OpenCV no puede abrirlo.")
        return

    print("Imagen cargada correctamente. Aplicando transformación de perspectiva...")

    # Coordenadas del panel en la imagen original (cuatro esquinas del panel)
    pts_origen = np.float32([[80, 19], [190, 35], [153, 162], [16, 140]])

    # Coordenadas destino: cuadrado de 224×224 px (tamaño de entrada de la red)
    pts_destino = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])

    # Calcular la matriz de homografía y aplicar la transformación
    matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
    img_aplanada = cv2.warpPerspective(img, matriz, (224, 224))

    # Dibujar el polígono del panel en la imagen original para visualización
    img_marcada = img.copy()
    cv2.polylines(
        img_marcada,
        [np.int32(pts_origen)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
    )

    # Guardar ambas imágenes en el directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_marcada.jpg"), img_marcada)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_aplanada.jpg"), img_aplanada)

    print("\nTransformación completada con éxito.")
    print(f"Revisa las imágenes en: {OUTPUT_DIR}")


if __name__ == "__main__":
    test_perspective_warp()
