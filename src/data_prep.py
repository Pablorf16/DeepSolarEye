

import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CATEGORY_BINS, CATEGORY_LABELS, DATA_SPLIT, RANDOM_STATE

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def parse_filename_regex(filename: str) -> dict:
    """Extract metadata from filename. Returns dict or None if parsing fails.
    
    Expected format: {panel_id}_L_{power_loss}_I_{irradiance}_{date}.jpg
    """


def process_and_split() -> None:
    """Pipeline: recursive extraction → metadata parsing → stratified split."""
    logger.info(f"Starting extraction from: {RAW_DATA_DIR}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    table_rows = []
    skipped = 0
    
    for path in RAW_DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            if not path.exists():
                logger.warning(f"File not found: {path}")
                continue
            
            row_data = parse_filename_regex(path.name)
            if row_data:
                row_data['filename'] = str(path.relative_to(RAW_DATA_DIR))
                table_rows.append(row_data)
            else:
                skipped += 1

    if not table_rows:
        raise FileNotFoundError(
            "No valid data found. Expected format: *_L_{power_loss}_I_{irradiance}*.jpg"
        )

    logger.info(f"Total images found: {len(table_rows)}, skipped: {skipped}")

    df = pd.DataFrame(table_rows)
    df = df[(df['power_loss'] >= 0) & (df['power_loss'] <= 100)]
    
    import numpy as np
    q25, q50, q75 = np.percentile(df['power_loss'], [25, 50, 75])
    
    quartile_bins = [-1, q25, q50, q75, 105]
    quartile_labels = ['Q1_Limpio', 'Q2_Moderado', 'Q3_Alto', 'Q4_Crítico']
    
    logger.info(f"\nComputed Quartiles:")
    logger.info(f"Q1 (0-25%):   {q25:.2f}%")
    logger.info(f"Q2 (25-50%):  {q50:.2f}%")
    logger.info(f"Q3 (50-75%):  {q75:.2f}%")
    logger.info(f"Q4 (75-100%): 100.00%\n")
    
    df['dirt_category'] = pd.cut(
        df['power_loss'],
        bins=quartile_bins,
        labels=quartile_labels,
        include_lowest=True
    )
    
    logger.info(f"Category Distribution (Original Dataset):\n{df['dirt_category'].value_counts().sort_index()}\n")

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
    
    logger.info(f"Split completed (balanced by quartiles).")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val:   {len(val_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train_dataset.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_dataset.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_dataset.csv", index=False)
    
    logger.info(f"\nData saved to {PROCESSED_DATA_DIR}")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val:   {len(val_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    logger.info(f"  Total: {len(train_df) + len(val_df) + len(test_df)} samples")


if __name__ == "__main__":
    process_and_split()



