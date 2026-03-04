"""
predict.py
==========
Módulo de inferencia para DeepSolarEye.

Carga el mejor modelo entrenado y predice la pérdida de potencia (%)
de una imagen de panel solar dada.

Uso::

    python src/predict.py --image <ruta_a_la_imagen> [--model <ruta_al_modelo>]

Ejemplo::

    python src/predict.py --image data/raw/panel_001.jpg
"""

import argparse
import os
import sys

import torch
from PIL import Image

# Añadir el directorio src al path para importar módulos locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_transforms  # noqa: E402
from model import Net               # noqa: E402

# ---------------------------------------------------------------------------
# Rutas por defecto (relativas a la raíz del proyecto)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'best_model.pth')

# Dispositivo de cómputo: GPU si está disponible, CPU en caso contrario
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path):
    """
    Carga el modelo ``Net`` desde un fichero de pesos ``.pth``.

    Parameters
    ----------
    model_path : str
        Ruta al fichero de pesos del modelo (``state_dict``).

    Returns
    -------
    Net
        Modelo cargado en modo evaluación y transferido al dispositivo
        correspondiente (CPU o GPU).

    Raises
    ------
    FileNotFoundError
        Si el fichero de pesos no existe en la ruta indicada.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en: {model_path}\n"
            "Asegúrate de haber ejecutado train.py primero."
        )

    model = Net()
    # map_location garantiza compatibilidad entre GPU y CPU
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()   # Desactiva dropout y BatchNorm en modo entrenamiento
    return model


def predict(image_path, model_path=DEFAULT_MODEL_PATH):
    """
    Predice la pérdida de potencia de un panel solar a partir de una imagen.

    Parameters
    ----------
    image_path : str
        Ruta a la imagen del panel solar (JPG, PNG, etc.).
    model_path : str, optional
        Ruta al fichero de pesos del modelo. Por defecto usa
        ``saved_models/best_model.pth``.

    Returns
    -------
    float
        Pérdida de potencia predicha en el rango [0, 100] %.

    Raises
    ------
    FileNotFoundError
        Si la imagen o el modelo no existen en las rutas indicadas.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    # Cargar modelo
    model = load_model(model_path)

    # Cargar y preprocesar imagen (mismas transformaciones que en evaluación)
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(phase='test')
    # unsqueeze(0) añade la dimensión de batch: (3, 224, 224) → (1, 3, 224, 224)
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inferencia sin calcular gradientes para ahorrar memoria
    with torch.no_grad():
        output = model(tensor)

    # El modelo devuelve un tensor (1, 1); extraemos el valor escalar
    power_loss = output.item()
    return power_loss


def main():
    """Punto de entrada con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="DeepSolarEye — Predicción de pérdida de potencia por suciedad"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help="Ruta a la imagen del panel solar a analizar.",
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Ruta al fichero de pesos del modelo (por defecto: {DEFAULT_MODEL_PATH}).",
    )
    args = parser.parse_args()

    try:
        result = predict(args.image, args.model)
        print(f"Imagen  : {args.image}")
        print(f"Pérdida de potencia predicha: {result:.2f} %")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
