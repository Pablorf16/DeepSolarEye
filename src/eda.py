# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_power_loss_distribution(csv_path: str, output_dir: str):
    """
    Genera un histograma de la distribución de la pérdida de potencia.
    Permite detectar desbalanceo de clases (sesgos) en el dataset.
    """
    # Carga de datos
    df = pd.read_csv(csv_path)
    
    target_col = 'power_loss' 
    if target_col not in df.columns:
        raise ValueError(f"La columna {target_col} no se encuentra en el CSV.")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.histplot(df[target_col], bins=50, kde=True, color='royalblue')
    
    plt.title('Distribución de la Pérdida de Potencia en el Dataset', fontsize=14)
    plt.xlabel('Pérdida de Potencia (%)', fontsize=12)
    plt.ylabel('Frecuencia (Número de Imágenes)', fontsize=12)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'power_loss_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Histograma de pérdida de potencia guardado en: {save_path}")

def plot_irradiance_distribution(csv_path: str, output_dir: str):
    """
    Genera un histograma de la irradiancia solar, si está disponible.
    """
    df = pd.read_csv(csv_path)
    if 'irradiance' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['irradiance'], bins=30, kde=True, color='darkorange')
        plt.title('Distribución de la Irradiancia Solar', fontsize=14)
        plt.xlabel('Irradiancia (W/m²)', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        
        save_path = os.path.join(output_dir, 'irradiance_histogram.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Histograma de irradiancia guardado en: {save_path}")
    else:
        print("[WARN] Columna de irradiancia no encontrada. Saltando gráfica.")

if __name__ == "__main__":
    # --- RUTAS CORREGIDAS ---
    # Subimos un nivel desde src/eda.py para llegar a la raíz del proyecto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Apuntamos exactamente a donde data_prep.py guardó los datos
    TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'train_dataset.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
    
    print("Iniciando Análisis Exploratorio de Datos (EDA)...")
    
    # Comprobación de seguridad antes de ejecutar
    if not os.path.exists(TRAIN_CSV):
        print(f"[ERROR] No se encuentra el archivo en: {TRAIN_CSV}")
        print("Asegúrate de haber ejecutado data_prep.py primero.")
    else:
        plot_power_loss_distribution(TRAIN_CSV, OUTPUT_DIR)
        plot_irradiance_distribution(TRAIN_CSV, OUTPUT_DIR)
        print("EDA completado con éxito.")