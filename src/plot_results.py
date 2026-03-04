import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_curves_v3(log_file, save_dir):
    """
    Genera gráficas de entrenamiento para v3.0
    
    Crea:
    1. RMSE vs Épocas (Train/Val)
    2. MAE y R² diagnósticos
    3. Learning Rate
    """
    if not os.path.exists(log_file):
        print(f"⚠️ No se encuentra: {log_file}")
        return False
    
    try:
        df = pd.read_csv(log_file)
        
        # Crear figura con subgráficos
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DeepSolarEye v3.0 - Análisis de Entrenamiento', fontsize=16, fontweight='bold', y=0.995)
        
        # ============== GRÁFICA 1: RMSE ==============
        ax1 = axes[0, 0]
        best_epoch = df.loc[df['val_rmse'].idxmin()]
        
        ax1.plot(df['epoch'], df['train_rmse'], label='Train RMSE', color='#1f77b4', linewidth=2, marker='o', markersize=3)
        ax1.plot(df['epoch'], df['val_rmse'], label='Val RMSE (Optimizing)', color='#ff7f0e', linewidth=2.5, marker='s', markersize=3)
        ax1.scatter(best_epoch['epoch'], best_epoch['val_rmse'], color='red', s=100, zorder=5, edgecolors='darkred', linewidth=2)
        ax1.annotate(f"Best: {best_epoch['val_rmse']:.3f}%\nÉpoca {int(best_epoch['epoch'])}", 
                     (best_epoch['epoch'], best_epoch['val_rmse']),
                     textcoords="offset points", xytext=(10, 10), 
                     ha='left', fontsize=9, fontweight='bold', color='red',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        ax1.set_xlabel('Épocas', fontsize=11)
        ax1.set_ylabel('RMSE (%)', fontsize=11)
        ax1.set_title('1. RMSE (Métrica de Optimización)', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=10)
        
        # ============== GRÁFICA 2: MAE ==============
        ax2 = axes[0, 1]
        ax2.plot(df['epoch'], df['val_mae'], label='Val MAE', color='#2ca02c', linewidth=2.5, marker='^', markersize=3)
        ax2.fill_between(df['epoch'], df['val_mae'], alpha=0.3, color='#2ca02c')
        ax2.set_xlabel('Épocas', fontsize=11)
        ax2.set_ylabel('MAE (%)', fontsize=11)
        ax2.set_title('2. MAE (Diagnóstico)', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=10)
        
        # ============== GRÁFICA 3: R² ==============
        ax3 = axes[1, 0]
        ax3.plot(df['epoch'], df['val_r2'], label='Val R²', color='#d62728', linewidth=2.5, marker='D', markersize=3)
        ax3.axhline(y=0.7, color='green', linestyle='--', linewidth=1, label='R² = 0.7 (Acceptable)', alpha=0.7)
        ax3.fill_between(df['epoch'], df['val_r2'], alpha=0.3, color='#d62728')
        ax3.set_xlabel('Épocas', fontsize=11)
        ax3.set_ylabel('R² (Coefficient)', fontsize=11)
        ax3.set_title('3. R² (Diagnóstico)', fontsize=12, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='lower right', fontsize=10)
        ax3.set_ylim([min(0, df['val_r2'].min()) - 0.1, 1.0])
        
        # ============== GRÁFICA 4: Learning Rate ==============
        ax4 = axes[1, 1]
        ax4.semilogy(df['epoch'], df['learning_rate'], label='Learning Rate', color='#9467bd', linewidth=2.5, marker='o', markersize=3)
        ax4.set_xlabel('Épocas', fontsize=11)
        ax4.set_ylabel('Learning Rate (log scale)', fontsize=11)
        ax4.set_title('4. Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.5, which='both')
        ax4.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Guardar figura
        os.makedirs(os.path.join(save_dir, '..', 'figures'), exist_ok=True)
        output_path = os.path.join(save_dir, '..', 'figures', 'training_curves_v3.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráficas guardadas en: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ Error generando gráficas: {e}")
        return False


def plot_learning_curve():
    """Función original para retrocompatibilidad con v2"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_FILE = os.path.join(BASE_DIR, 'training_log.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
    
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] No se encuentra el archivo {LOG_FILE}.")
        return

    df = pd.read_csv(LOG_FILE)
    
    # AHORA BUSCAMOS EL MEJOR MODELO EN VALIDACIÓN (Rigor científico)
    best_epoch = df.loc[df['val_loss'].idxmin()]
    
    plt.figure(figsize=(10, 6))
    
    # GRAFICAMOS TRAIN VS VAL
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss (MSE)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss (MSE)', color='#2ca02c', linewidth=2, marker='s', markersize=4)
    
    plt.scatter(best_epoch['epoch'], best_epoch['val_loss'], color='red', s=100, zorder=5, label=f"Mejor Modelo (Época {int(best_epoch['epoch'])})")
    plt.annotate(f"{best_epoch['val_loss']:.2f}", 
                 (best_epoch['epoch'], best_epoch['val_loss']),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', fontsize=10, fontweight='bold', color='red')

    plt.title('Curva de Aprendizaje - DeepSolarEye V2', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Épocas (Epochs)', fontsize=12)
    plt.ylabel('Error Cuadrático Medio (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10, shadow=True)
    
    plt.xticks(range(1, int(df['epoch'].max()) + 1))
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'learning_curve_v2.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ ¡Gráfica V2 generada y guardada en: {output_path}")

if __name__ == "__main__":
    plot_learning_curve()