"""
data_prep.py - Preparación de datos para DeepSolarEye v3.0

Pipeline de extracción, validación y división estratificada de datos.

Entrada: Imágenes RAW en data/raw/Solar_Panel_Soiling_Image_dataset/PanelImages/
         con nombres como: *_L_{power_loss}_I_{irradiance}_{date}

Salida: CSVs estratificados:
  - data/processed/train_dataset.csv (60%, oversampleado)
  - data/processed/val_dataset.csv (20%, original)
  - data/processed/test_dataset.csv (20%, original)
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Importar configuración centralizada
from config import CATEGORY_BINS, CATEGORY_LABELS, RANDOM_STATE

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Rutas dinámicas
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def parse_filename_regex(filename: str) -> dict:
    """
    Extrae metadata desde el nombre de archivo usando Regex.
    
    FORMATO ESPERADO DE NOMBRE:
    ──────────────────────────
    {panel_id}_L_{power_loss}_I_{irradiance}_{date}.jpg
    
    Ejemplo: panel_001_L_12.5_I_800_01Jan2021.jpg
    
    CAMPOS EXTRAÍDOS:
    - power_loss: Pérdida de potencia (%) extraída de "_L_{value}"
    - irradiance: Irradiancia solar (W/m²) extraída de "_I_{value}"
    - date: Fecha de captura extraída con patrón de año (201X)
    
    Args:
        filename (str): Nombre de archivo con extensión
    
    Returns:
        dict: Diccionario con claves [filename, date, irradiance, power_loss]
              Si falla la extracción, retorna None (será filtrado)
    
    Raises:
        None: Las excepciones son capturadas e loguean sin interrumpir
    
    Example:
        >>> result = parse_filename_regex('panel_001_L_12.5_I_800_01Jan2021.jpg')
        >>> result
        {
            'filename': 'panel_001_L_12.5_I_800_01Jan2021.jpg',
            'date': '01Jan2021',
            'irradiance': 800.0,
            'power_loss': 1250.0  # Nota: se multiplica por 100 para escala 0-100%
        }
    """
    # Eliminar extensiones para procesar nombre base
    clean_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    
    metadata = {
        'filename': filename,
        'date': None,
        'irradiance': 0.0,
        'power_loss': 0.0
    }

    try:
        # 1. EXTRACCIÓN DE PÉRDIDA DE POTENCIA (REQUERIDA)
        # Patrón: _L_{number} (ej: _L_12.5 o _L_5)
        match_loss = re.search(r'_L_([0-9\.]+)', clean_name)
        if match_loss:
            # Convertir a porcentaje (escala 0-100)
            raw_val = float(match_loss.group(1))
            metadata['power_loss'] = raw_val * 100
        else:
            # Sin power_loss: no se puede crear label → skip
            return None

        # 2. EXTRACCIÓN DE IRRADIANCIA (OPCIONAL)
        # Patrón: _I_{number} (ej: _I_800 o _I_950.5)
        match_irr = re.search(r'_I_([0-9\.]+)', clean_name)
        if match_irr:
            metadata['irradiance'] = float(match_irr.group(1))

        # 3. EXTRACCIÓN DE FECHA (OPCIONAL)
        # Patrón: {day}{month}_{year} donde year = 201X o 202X
        match_year = re.search(r'([A-Za-z]{3}_[A-Za-z]{3}_\d+.*?20[12]\d)', clean_name)
        if match_year:
            metadata['date'] = match_year.group(1)
        else:
            metadata['date'] = "Unknown"

        return metadata

    except Exception as e:
        logger.warning(f"Error parseando {filename}: {e}")
        return None


def oversample_dataframe(df: pd.DataFrame, stratify_col: str = 'dirt_category') -> pd.DataFrame:
    """
    Realiza oversampling equilibrando categorías minoritarias.
    
    JUSTIFICACIÓN (Plan v3.0 Punto 6):
    ──────────────────────────────────
    WeightedRandomSampler solo re-pondera durante muestreo (no crea variabilidad).
    Oversampling físico DUPLICA filas de clases minoritarias en el CSV.
    Combinado con Data Augmentation en train.py → máximas ganancias.
    
    ESTRATEGIA:
    1. Encuentra clase mayoritaria (contar máximo)
    2. Para cada categoría:
       - Si minoritaria: samplea CON reemplazo hasta igualar mayoritaria
       - Si mayoritaria: mantiene tal cual
    3. Shufflea resultado para evitar clustering
    
    RESULTADO:
    - Todas las categorías tienen equal representation en el CSV
    - Índices shuffleados para aleatoriedad
    
    Args:
        df (pd.DataFrame): DataFrame con columna de categoría
        stratify_col (str): Nombre de la columna categórica (default: 'dirt_category')
    
    Returns:
        pd.DataFrame: DataFrame expandido con filas duplicadas intelligentemente
    
    Example:
        >>> df_original = pd.DataFrame({'value': [1,2,3], 'cat': ['A','A','B']})
        >>> # 'A': 2 muestras, 'B': 1 muestra
        >>> df_balanced = oversample_dataframe(df_original, 'cat')
        >>> # 'A': 2 muestras, 'B': 2 muestras (oversampleado)
    """
    # Encontrar clase mayoritaria
    max_count = df[stratify_col].value_counts().max()
    
    # REPORTE: Distribución ANTES del oversampling
    logger.info("\n📊 DISTRIBUCIÓN ANTES DEL OVERSAMPLING:")
    for label in CATEGORY_LABELS:
        count = (df[stratify_col] == label).sum()
        pct = 100 * count / len(df) if len(df) > 0 else 0
        logger.info(f"   {label:12s}: {count:4d} ({pct:5.1f}%)")
    
    # Oversamplear: duplicar filas de clases minoritarias
    dfs_oversampled = []
    for label in CATEGORY_LABELS:
        class_df = df[df[stratify_col] == label]
        
        if len(class_df) < max_count:
            # Clase minoritaria: samplear con reemplazo hasta max_count
            oversampled = class_df.sample(
                n=max_count,
                replace=True,  # Permite replicar filas
                random_state=RANDOM_STATE
            )
            dfs_oversampled.append(oversampled)
        else:
            # Clase mayoritaria: mantener tal cual
            dfs_oversampled.append(class_df)
    
    # Concatenar y shufflear para evitar que oversample sea contíguo
    df_balanced = pd.concat(dfs_oversampled, ignore_index=True)
    df_balanced = df_balanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)
    
    # REPORTE: Distribución DESPUÉS del oversampling
    logger.info("\n📊 DISTRIBUCIÓN DESPUÉS DEL OVERSAMPLING:")
    for label in CATEGORY_LABELS:
        count = (df_balanced[stratify_col] == label).sum()
        pct = 100 * count / len(df_balanced) if len(df_balanced) > 0 else 0
        logger.info(f"   {label:12s}: {count:4d} ({pct:5.1f}%)")
    
    expansion_pct = 100 * (len(df_balanced) - len(df)) / len(df)
    logger.info(
        f"\n✅ Dataset expandido: {len(df)} → {len(df_balanced)} "
        f"ejemplos (+{expansion_pct:.1f}%)"
    )
    
    return df_balanced


def process_and_split() -> None:
    """
    Pipeline completo: extracción → validación → split → oversample.
    
    Pasos:
    1. Barrido recursivo de RAW_DATA_DIR buscando imágenes
    2. Extracción de metadata con regex
    3. Validación: power_loss ∈ [0, 100]
    4. Creación de categorías (Limpio/Leve/Moderado/Alto/Crítico)
    5. Split estratificado: 60% train, 20% val, 20% test
    6. Oversampling SOLO en train set
    7. Guardado de CSVs
    """
    logger.info(f"Iniciando extracción con REGEX en: {RAW_DATA_DIR}")
    
    # Extensiones de imagen válidas (case-insensitive)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    table_rows = []
    skipped = 0
    
    # BARRIDO RECURSIVO de imágenes
    for path in RAW_DATA_DIR.rglob('*'):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            
            # Validación robusta: archivo existe
            if not path.exists():
                logger.warning(f"Archivo fantasma detectado: {path}")
                continue
            
            # Extracción con regex
            row_data = parse_filename_regex(path.name)
            if row_data:
                # Guardar ruta RELATIVA desde RAW_DATA_DIR (para portabilidad)
                row_data['filename'] = str(path.relative_to(RAW_DATA_DIR))
                table_rows.append(row_data)
            else:
                skipped += 1

    # Error si no hay datos válidos
    if not table_rows:
        raise FileNotFoundError(
            "ERROR: No se encontraron datos válidos. "
            "Verifica que los archivos sigan el formato: "
            "*_L_{power_loss}_I_{irradiance}*.jpg"
        )

    logger.info(
        f"Total imágenes encontradas: {len(table_rows)}, saltadas: {skipped}"
    )

    # CREACIÓN DE DATAFRAME
    df = pd.DataFrame(table_rows)
    
    # Validación: power_loss ∈ [0, 100]%
    df = df[(df['power_loss'] >= 0) & (df['power_loss'] <= 100)]
    
    # ESTRATIFICACIÓN: Crear categorías de suciedad
    # (explicadas exhaustivamente en config.py)
    df['dirt_category'] = pd.cut(
        df['power_loss'],
        bins=CATEGORY_BINS,
        labels=CATEGORY_LABELS,
        include_lowest=True
    )
    
    logger.info("\n=== DISTRIBUCIÓN DE CATEGORÍAS (Dataset Original) ===")
    logger.info(f"\n{df['dirt_category'].value_counts().sort_index()}\n")

    # DIVISIÓN ESTRATIFICADA: 60% train, 20% val, 20% test
    # Método: train_test_split con estratificación para mantener
    # proporciones de categorías en cada split
    logger.info("🔀 DIVIDIENDO DATASET (60% train, 20% val, 20% test)...")
    
    # Paso 1: Separar 60% para train, 40% para temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,  # 40% para val+test
        random_state=RANDOM_STATE,
        stratify=df['dirt_category']
    )
    
    # Paso 2: Dividir el 40% restante en 50/50 para val y test
    # (50% del 40% = 20% total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,  # 50% del 40% = 20% total
        random_state=RANDOM_STATE,
        stratify=temp_df['dirt_category']
    )
    
    logger.info(f"✅ Split realizado:")
    logger.info(f"   Train (antes oversample): {len(train_df)} muestras")
    logger.info(f"   Val:  {len(val_df)} muestras")
    logger.info(f"   Test: {len(test_df)} muestras")
    
    # OVERSAMPLING: SOLO en train_df (después del split)
    # NOTA CRÍTICA: Oversample DESPUÉS del split previene data leakage
    logger.info("\n🌊 APLICANDO OVERSAMPLING AL TRAIN SET...")
    train_df = oversample_dataframe(train_df, stratify_col='dirt_category')
    
    # GUARDADO: Crear directorio y guardar CSVs
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train_dataset.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val_dataset.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_dataset.csv", index=False)
    
    logger.info(f"\n✅ Datos guardados en {PROCESSED_DATA_DIR}")
    logger.info(f"   Train samples (oversampleado): {len(train_df)}")
    logger.info(f"   Val samples:   {len(val_df)}")
    logger.info(f"   Test samples:  {len(test_df)}")
    logger.info(f"   Total:         {len(train_df) + len(val_df) + len(test_df)}")


if __name__ == "__main__":
    process_and_split()
