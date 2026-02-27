import pandas as pd
import re
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

def parse_filename_regex(filename):
    """
    Extrae datos usando Expresiones Regulares (Regex) para máxima precisión.
    """
    clean_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    
    metadata = {
        'filename': filename,
        'date': None,
        'irradiance': 0.0,
        'power_loss': 0.0
    }

    try:
        # 1. Extracción de PÉRDIDA DE POTENCIA (Target)
        match_loss = re.search(r'_L_([0-9\.]+)', clean_name)
        if match_loss:
            raw_val = float(match_loss.group(1))
            metadata['power_loss'] = raw_val * 100
        else:
            return None

        # 2. Extracción de IRRADIANCIA (Input auxiliar)
        match_irr = re.search(r'_I_([0-9\.]+)', clean_name)
        if match_irr:
            metadata['irradiance'] = float(match_irr.group(1))

        # 3. Extracción de FECHA (Metadata)
        match_year = re.search(r'([A-Za-z]{3}_[A-Za-z]{3}_\d+.*?201\d)', clean_name)
        if match_year:
            metadata['date'] = match_year.group(1)
        else:
            metadata['date'] = "Unknown"

        return metadata

    except Exception as e:
        print(f"Error parseando {filename}: {e}")
        return None

def process_and_split():
    logger.info(f"Iniciando extracción con REGEX en: {RAW_DATA_DIR}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    table_rows = []
    skipped = 0
    
    # BARRIDO DE IMÁGENES
    for path in RAW_DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            
            if not path.exists():
                logger.warning(f"Archivo fantasma detectado (imposible): {path}")
                continue
                
            row_data = parse_filename_regex(path.name)
            if row_data:
                row_data['filename'] = str(path.relative_to(RAW_DATA_DIR))
                table_rows.append(row_data)
            else:
                skipped += 1

    if not table_rows:
        raise FileNotFoundError("ERROR: No se encontraron datos válidos. Verifica que los archivos sigan el formato: *_L_{power_loss}_I_{irradiance}*.jpg")

    logger.info(f"Total imágenes encontradas: {len(table_rows)}, saltadas: {skipped}")

    # CREACIÓN DE DATAFRAME
    df = pd.DataFrame(table_rows)
    df = df[(df['power_loss'] >= 0) & (df['power_loss'] <= 100)]
    
    
    #ESTRATIFICACIÓN PARA BALANCEO DE CLASES
   
    # Creamos tramos de pérdida de potencia para forzar una división justa
    bins = [-1, 5, 15, 30, 60, 105]
    labels = ['Limpio (0-5%)', 'Leve (5-15%)', 'Moderado (15-30%)', 'Alto (30-60%)', 'Critico (60-100%)']
    
    # Asignamos a cada imagen su tramo correspondiente
    df['dirt_category'] = pd.cut(df['power_loss'], bins=bins, labels=labels)
    
    logger.info("=== DISTRIBUCIÓN DE CATEGORÍAS ===")
    logger.info(f"\n{df['dirt_category'].value_counts()}\n")

    # DIVISIÓN ESTRATIFICADA EN 3 SETS (60% train, 20% val, 20% test)
    # Esto evita data leakage en early stopping
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.40,  # 40% para val+test
        random_state=42, 
        stratify=df['dirt_category']
    )
    
    # Split del 40% restante en 50/50 para val y test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,  # 50% del 40% = 20% total
        random_state=42,
        stratify=temp_df['dirt_category']
    )
    
    # GUARDADO
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train_dataset.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_dataset.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_dataset.csv", index=False)
    
    logger.info(f"✅ Datos guardados en {PROCESSED_DATA_DIR}")
    logger.info(f"   Train samples: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    logger.info(f"   Val samples:   {len(val_df)} ({100*len(val_df)/len(df):.1f}%)")
    logger.info(f"   Test samples:  {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")
    logger.info(f"   Total:         {len(df)}")

if __name__ == "__main__":
    process_and_split()