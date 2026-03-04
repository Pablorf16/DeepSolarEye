"""
data_prep.py
============
Módulo de preparación de datos para DeepSolarEye.

Extrae metadatos de los nombres de fichero de imagen (pérdida de potencia,
irradiancia y fecha) mediante expresiones regulares y genera tres CSVs
estratificados (train / val / test) en data/processed/.

Formato esperado de nombre de fichero:
    *_L_{power_loss}_I_{irradiance}_{fecha}.jpg
    Ejemplo: solar_Fri_Jun_16_L_0.12_I_0.003.jpg
"""

import re
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración del logger
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas del proyecto (relativas a este fichero)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def parse_filename_regex(filename):
    """
    Extrae metadatos de un nombre de fichero usando expresiones regulares.

    Parameters
    ----------
    filename : str
        Nombre del fichero de imagen (p. ej. ``solar_L_0.09_I_0.003.jpg``).

    Returns
    -------
    dict or None
        Diccionario con las claves ``filename``, ``date``, ``irradiance`` y
        ``power_loss``, o ``None`` si el nombre no cumple el formato esperado.
    """
    # Eliminamos la extensión para simplificar el parsing
    clean_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')

    metadata = {
        'filename': filename,
        'date': None,
        'irradiance': 0.0,
        'power_loss': 0.0,
    }

    try:
        # 1. Extraer PÉRDIDA DE POTENCIA (variable objetivo / target)
        match_loss = re.search(r'_L_([0-9\.]+)', clean_name)
        if match_loss:
            # El valor en el fichero está en escala 0-1; lo convertimos a %
            raw_val = float(match_loss.group(1))
            metadata['power_loss'] = raw_val * 100
        else:
            # Si no se encuentra el campo obligatorio, descartamos el fichero
            return None

        # 2. Extraer IRRADIANCIA solar (variable auxiliar de entrada)
        match_irr = re.search(r'_I_([0-9\.]+)', clean_name)
        if match_irr:
            metadata['irradiance'] = float(match_irr.group(1))

        # 3. Extraer FECHA de captura (metadato informativo)
        match_year = re.search(r'([A-Za-z]{3}_[A-Za-z]{3}_\d+.*?201\d)', clean_name)
        metadata['date'] = match_year.group(1) if match_year else "Unknown"

        return metadata

    except Exception as e:
        logger.error(f"Error parseando {filename}: {e}")
        return None


def process_and_split():
    """
    Recorre RAW_DATA_DIR, extrae metadatos de cada imagen y genera los CSVs
    de entrenamiento, validación y test con división estratificada 60/20/20.

    Los ficheros resultantes se guardan en PROCESSED_DATA_DIR:
        - train_dataset.csv
        - val_dataset.csv
        - test_dataset.csv

    Raises
    ------
    FileNotFoundError
        Si no se encuentra ningún fichero de imagen válido en RAW_DATA_DIR.
    """
    logger.info(f"Iniciando extracción con REGEX en: {RAW_DATA_DIR}")

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    table_rows = []
    skipped = 0

    # Barrido recursivo de imágenes en el directorio raw
    for path in RAW_DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            row_data = parse_filename_regex(path.name)
            if row_data:
                # Guardamos la ruta relativa para que sea portátil
                row_data['filename'] = str(path.relative_to(RAW_DATA_DIR))
                table_rows.append(row_data)
            else:
                skipped += 1

    if not table_rows:
        raise FileNotFoundError(
            "ERROR: No se encontraron datos válidos. "
            "Verifica que los archivos sigan el formato: "
            "*_L_{power_loss}_I_{irradiance}*.jpg"
        )

    logger.info(f"Total imágenes encontradas: {len(table_rows)}, saltadas: {skipped}")

    # Crear DataFrame y filtrar valores de pérdida fuera del rango [0, 100]
    df = pd.DataFrame(table_rows)
    df = df[(df['power_loss'] >= 0) & (df['power_loss'] <= 100)]

    # Estratificación: asignamos cada imagen a un tramo de suciedad
    # para garantizar una división justa entre los tres conjuntos
    bins = [-1, 5, 15, 30, 60, 105]
    labels = [
        'Limpio (0-5%)',
        'Leve (5-15%)',
        'Moderado (15-30%)',
        'Alto (30-60%)',
        'Critico (60-100%)',
    ]
    df['dirt_category'] = pd.cut(df['power_loss'], bins=bins, labels=labels)

    logger.info("=== DISTRIBUCIÓN DE CATEGORÍAS ===")
    logger.info(f"\n{df['dirt_category'].value_counts()}\n")

    # División estratificada en 3 conjuntos: 60% train, 20% val, 20% test
    # Usar validación separada evita data leakage en el early stopping
    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,       # 40% para val + test
        random_state=42,
        stratify=df['dirt_category'],
    )

    # El 40% restante se divide al 50/50 → 20% val y 20% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,       # 50% del 40% = 20% del total
        random_state=42,
        stratify=temp_df['dirt_category'],
    )

    # Guardar los tres CSVs en el directorio procesado
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train_dataset.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_dataset.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_dataset.csv", index=False)

    logger.info(f"Datos guardados en {PROCESSED_DATA_DIR}")
    logger.info(f"   Train samples: {len(train_df)} ({100 * len(train_df) / len(df):.1f}%)")
    logger.info(f"   Val samples:   {len(val_df)} ({100 * len(val_df) / len(df):.1f}%)")
    logger.info(f"   Test samples:  {len(test_df)} ({100 * len(test_df) / len(df):.1f}%)")
    logger.info(f"   Total:         {len(df)}")


if __name__ == "__main__":
    process_and_split()
