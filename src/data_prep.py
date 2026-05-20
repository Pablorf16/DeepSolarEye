
"""
Preparación y estratificación de datos para DeepSolarEye v4.0.

Ejecuta el pipeline de preparación:
1. Extrae imágenes recursivamente desde data/raw
2. Parsea poder de pérdida e irradiancia del nombre de archivo
3. Calcula cuartiles dinámicos del dataset (Q1, Q2, Q3, Q4)
4. Divide en train/val/test estratificado por cuartiles (60/20/20)
5. Guarda CSVs procesados y persiste cuartiles en JSON

Se ejecuta una única vez como script de preparación inicial.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CATEGORY_LABELS, DATA_SPLIT, RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
QUARTILES_FILE = BASE_DIR / "data" / "quartiles.json"


def parse_filename_regex(filename: str) -> dict:
    """Extrae poder de pérdida e irradiancia del nombre de archivo usando patrón regex."""
    pattern = r'_L_([0-9.]+)_I_([0-9.]+)'
    match = re.search(pattern, filename)
    if match:
        try:
            return {
                'power_loss': float(match.group(1)),
                'irradiance': float(match.group(2))
            }
        except ValueError:
            return None
    return None


def process_and_split() -> None:
    """Pipeline: extracción recursiva → parseo de metadatos → división estratificada."""
    logger.info("Extracting images from: %s", RAW_DATA_DIR)
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    table_rows = []
    skipped = 0

    # Buscar archivos recursivamente
    for path in RAW_DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            row_data = parse_filename_regex(path.name)
            if row_data:
                row_data['filename'] = str(path.relative_to(RAW_DATA_DIR))
                table_rows.append(row_data)
            else:
                skipped += 1

    if not table_rows:
        raise FileNotFoundError(
            "No valid data found. Expected format: *_L_{loss}_I_{irr}*.jpg"
        )

    logger.info("Images found: %d, skipped: %d", len(table_rows), skipped)

    df = pd.DataFrame(table_rows)
    # Filtrar rango válido de power_loss
    df = df[(df['power_loss'] >= 0) & (df['power_loss'] <= 100)]

    # Calcular cuartiles dinámicos del dataset actual
    q25, q50, q75 = np.percentile(df['power_loss'], [25, 50, 75])

    quartile_bins = [-1, q25, q50, q75, 105]
    quartile_labels = ['Q1_Limpio', 'Q2_Moderado', 'Q3_Alto', 'Q4_Crítico']

    logger.info("\nDynamic Quartiles:")
    logger.info("Q1 (0-25%%):   %.2f%%", q25)
    logger.info("Q2 (25-50%%):  %.2f%%", q50)
    logger.info("Q3 (50-75%%):  %.2f%%", q75)
    logger.info("Q4 (75-100%%): 100.00%%\n")

    # Categorizar por cuartiles
    df['dirt_category'] = pd.cut(
        df['power_loss'],
        bins=quartile_bins,
        labels=quartile_labels,
        include_lowest=True
    )

    logger.info(
        "Category Distribution:\n%s\n",
        df['dirt_category'].value_counts().sort_index()
    )

    # Dividir en train/val/test estratificado
    logger.info("Splitting dataset (60% train, 20% val, 20% test)...")

    train_df, temp_df = train_test_split(
        df,
        test_size=(DATA_SPLIT['val'] + DATA_SPLIT['test']),
        random_state=RANDOM_STATE,
        stratify=df['dirt_category']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=DATA_SPLIT['test'] / (DATA_SPLIT['val'] + DATA_SPLIT['test']),
        random_state=RANDOM_STATE,
        stratify=temp_df['dirt_category']
    )

    logger.info("Split completed (balanced by quartiles).")
    logger.info("  Train: %d samples", len(train_df))
    logger.info("  Val:   %d samples", len(val_df))
    logger.info("  Test:  %d samples", len(test_df))

    # Guardar CSVs procesados
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train_dataset.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_dataset.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_dataset.csv", index=False)

    # Persistir cuartiles para reproducibilidad (dataset.py)
    quartiles_data = {
        'q25': float(q25),
        'q50': float(q50),
        'q75': float(q75),
        'bins': quartile_bins,
        'labels': quartile_labels
    }
    with open(QUARTILES_FILE, 'w', encoding='utf-8') as f:
        json.dump(quartiles_data, f, indent=2)
    
    logger.info("\nData saved to %s", PROCESSED_DATA_DIR)
    logger.info("  Total: %d samples", len(train_df) + len(val_df) + len(test_df))
    logger.info("Quartiles saved to %s", QUARTILES_FILE)


if __name__ == "__main__":
    process_and_split()



