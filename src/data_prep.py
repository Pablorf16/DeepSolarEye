import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

def parse_filename_regex(filename):
    """
    Extrae datos usando Expresiones Regulares (Regex) para máxima precisión.
    Formato detectado: solar_..._L_0.1539..._I_0.8264...jpg
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
        # Buscamos "_L_" seguido de números y puntos
        match_loss = re.search(r'_L_([0-9\.]+)', clean_name)
        if match_loss:
            # El valor viene en tanto por uno (ej: 0.1539). 
            # Lo pasamos a porcentaje (15.39%) para que sea más legible.
            raw_val = float(match_loss.group(1))
            metadata['power_loss'] = raw_val * 100
        else:
            return None # Si no tiene etiqueta L, no nos sirve para entrenar

        # 2. Extracción de IRRADIANCIA (Input auxiliar)
        # Buscamos "_I_" seguido de números
        match_irr = re.search(r'_I_([0-9\.]+)', clean_name)
        if match_irr:
            metadata['irradiance'] = float(match_irr.group(1))

        # 3. Extracción de FECHA (Metadata)
        # Buscamos el año 2016 o 2017 para ubicar la fecha
        # Formato: solar_Dia_Mes_Num_..._Año
        # Ejemplo: solar_Wed_Jun_21_13__7__33_2017
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
    print(f"--> [ETL] Iniciando extracción con REGEX en: {RAW_DATA_DIR}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    table_rows = []
    
    # BARRIDO DE IMÁGENES
    for path in RAW_DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            row_data = parse_filename_regex(path.name)
            
            if row_data:
                # Ruta relativa
                row_data['filename'] = str(path.relative_to(RAW_DATA_DIR))
                table_rows.append(row_data)

    if not table_rows:
        raise FileNotFoundError("ERROR: No se encontraron datos válidos.")

    # CREACIÓN DE DATAFRAME
    df = pd.DataFrame(table_rows)
    
    # LIMPIEZA
    # Aseguramos que el porcentaje esté entre 0 y 100
    df = df[(df['power_loss'] >= 0) & (df['power_loss'] <= 100)]
    
    print("\n=== MUESTRA DE LA TABLA REAL (CORREGIDA) ===")
    print(df[['filename', 'date', 'irradiance', 'power_loss']].head())
    print("============================================\n")

    # DIVISIÓN Y GUARDADO
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train_dataset.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_dataset.csv", index=False)
    
    print(f"--> ¡ÉXITO! Datos guardados en {PROCESSED_DATA_DIR}")
    print(f"    Total Train: {len(train_df)}")
    print(f"    Total Test:  {len(test_df)}")

if __name__ == "__main__":
    process_and_split()