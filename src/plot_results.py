import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats


def plot_training_curves_v3(log_file, save_dir):
    """Genera gráficas de entrenamiento con métricas clave.
    
    Crea: RMSE, MAE, R², Learning Rate
    
    Args:
        log_file: Ruta al CSV de training_log_v3.csv
        save_dir: Directorio donde guardar figuras
        
    Returns:
        bool: True si éxito, False si error
    """
    if not os.path.exists(log_file):
        print(f"Error: No se encuentra {log_file}")
        return False
    
    try:
        # Intentar leer con diferentes encodings
        df = None
        for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(log_file, encoding=encoding)
                break
            except:
                continue
        
        if df is None:
            print(f"Error: No se pudo leer {log_file}")
            return False
        
        # Crear figura con subgráficos
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Analysis - Model Performance', fontsize=16, fontweight='bold', y=0.995)
        
        # RMSE plot
        ax1 = axes[0, 0]
        best_epoch = df.loc[df['val_rmse'].idxmin()]
        
        ax1.plot(df['epoch'], df['train_rmse'], label='Train RMSE', color='#1f77b4', linewidth=2, marker='o', markersize=3)
        ax1.plot(df['epoch'], df['val_rmse'], label='Val RMSE (Optimizing)', color='#ff7f0e', linewidth=2.5, marker='s', markersize=3)
        ax1.scatter(best_epoch['epoch'], best_epoch['val_rmse'], color='red', s=100, zorder=5, edgecolors='darkred', linewidth=2)
        
        ax1.set_xlabel('Episodios', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSE (%)', fontsize=14, fontweight='bold')
        ax1.set_title('RMSE (Optimization Metric)', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=10)
        
        # MAE plot
        ax2 = axes[0, 1]
        ax2.plot(df['epoch'], df['val_mae'], label='Val MAE', color='#2ca02c', linewidth=2.5, marker='^', markersize=3)
        ax2.fill_between(df['epoch'], df['val_mae'], alpha=0.3, color='#2ca02c')
        ax2.set_xlabel('Episodios', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE (%)', fontsize=14, fontweight='bold')
        ax2.set_title('MAE (Diagnostic)', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=10)
        
        # R² plot
        ax3 = axes[1, 0]
        ax3.plot(df['epoch'], df['val_r2'], label='Val R²', color='#d62728', linewidth=2.5, marker='D', markersize=3)
        ax3.axhline(y=0.7, color='green', linestyle='--', linewidth=1, label='R² = 0.7 (Acceptable)', alpha=0.7)
        ax3.fill_between(df['epoch'], df['val_r2'], alpha=0.3, color='#d62728')
        ax3.set_xlabel('Episodios', fontsize=14, fontweight='bold')
        ax3.set_ylabel('R² (Coefficient)', fontsize=14, fontweight='bold')
        ax3.set_title('R² (Diagnostic)', fontsize=12, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='lower right', fontsize=10)
        ax3.set_ylim([min(0, df['val_r2'].min()) - 0.1, 1.0])
        
        # Learning rate plot
        ax4 = axes[1, 1]
        ax4.semilogy(df['epoch'], df['learning_rate'], label='Learning Rate', color='#9467bd', linewidth=2.5, marker='o', markersize=3)
        ax4.set_xlabel('Episodios', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Learning Rate (log scale)', fontsize=14, fontweight='bold')
        ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.5, which='both')
        ax4.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Guardar figura
        os.makedirs(os.path.join(save_dir, '..', 'figures'), exist_ok=True)
        output_path = os.path.join(save_dir, '..', 'figures', 'training_curves_v3.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Info: Training curves saved in {output_path}")
        return True
        
    except Exception as e:
        print(f"Error generating training curves: {e}")
        return False


def plot_predictions_vs_reference(y_true, y_pred, test_df, save_dir):
    """Genera visualizaciones de predicción vs referencia para análisis de precisión.
    
    Crea scatter plot, residuals analysis, performance por categoría y error range analysis.
    
    Args:
        y_true: Labels verdaderos del test set
        y_pred: Predicciones del modelo
        test_df: DataFrame con columna 'dirt_category'
        save_dir: Directorio donde guardar figuras
    
    Returns:
        bool: True si éxito, False si error
    """
    os.makedirs(os.path.join(save_dir, '..', 'figures'), exist_ok=True)
    output_dir = os.path.join(save_dir, '..', 'figures')
    
    try:
        # Scatter plot: Prediction vs Reference
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {
            'Q1_Limpio': '#2ecc71',
            'Q2_Moderado': '#f39c12',
            'Q3_Alto': '#e74c3c',
            'Q4_Crítico': '#8b0000'
        }
        
        for category in test_df['dirt_category'].unique():
            if pd.isna(category):
                continue
            mask = test_df['dirt_category'] == category
            ax.scatter(y_true[mask], y_pred[mask], alpha=0.6, 
                      label=category, s=50, color=colors.get(category, 'gray'))
        
        # Línea perfecta y=x
        min_val, max_val = 0, 100
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, 
               label='Perfect Prediction', alpha=0.5)
        
        ax.set_xlabel('Referencia (Power Loss %)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicción (Power Loss %)', fontsize=12, fontweight='bold')
        ax.set_title('Prediction vs Reference (Test Set)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        plt.tight_layout()
        scatter_path = os.path.join(output_dir, 'scatter_pred_vs_ref.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: Scatter plot - {scatter_path}")
        
        # Residuals analysis
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        ax = axes[0, 0]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Error (y_pred - y_true)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Residuals Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Residuals vs Reference
        ax = axes[0, 1]
        ax.scatter(y_true, residuals, alpha=0.6, s=50)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Reference (Power Loss %)', fontsize=11)
        ax.set_ylabel('Residual', fontsize=11)
        ax.set_title('Residuals vs Reference', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Absolute Error per Category
        ax = axes[1, 0]
        abs_errors = np.abs(residuals)
        df_temp = pd.DataFrame({
            'Error': abs_errors,
            'Category': test_df['dirt_category']
        })
        df_temp.boxplot(column='Error', by='Category', ax=ax)
        ax.set_ylabel('Absolute Error (%)', fontsize=11)
        ax.set_title('Error by Category', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.sca(ax)
        plt.xticks(rotation=15, ha='right')
        
        # Q-Q plot
        ax = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        residuals_path = os.path.join(output_dir, 'residuals_analysis.png')
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: Residuals analysis - {residuals_path}")
        
        # Performance by Category
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        categories = sorted(test_df['dirt_category'].dropna().unique())
        rmse_list = []
        mae_list = []
        
        for cat in categories:
            mask = test_df['dirt_category'] == cat
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
            mae = np.mean(np.abs(y_t - y_p))
            
            rmse_list.append(rmse)
            mae_list.append(mae)
        
        colors_list = ['#2ecc71', '#f39c12', '#e74c3c', '#8b0000']
        
        # RMSE
        ax = axes[0]
        ax.bar(range(len(categories)), rmse_list, color=colors_list[:len(categories)], 
               edgecolor='black', alpha=0.7)
        ax.set_ylabel('RMSE (%)', fontsize=11, fontweight='bold')
        ax.set_title('RMSE by Category (Test Set)', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(rmse_list):
            ax.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=9)
        
        # MAE
        ax = axes[1]
        ax.bar(range(len(categories)), mae_list, color=colors_list[:len(categories)], 
               edgecolor='black', alpha=0.7)
        ax.set_ylabel('MAE (%)', fontsize=11, fontweight='bold')
        ax.set_title('MAE by Category (Test Set)', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mae_list):
            ax.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        perf_path = os.path.join(output_dir, 'performance_by_category.png')
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: Performance by category - {perf_path}")
        
        # Statistical summary
        print(f"\nGlobal Metrics (Test Set):")
        print(f"   RMSE: {np.sqrt(np.mean((y_true - y_pred)**2)):.4f}%")
        print(f"   MAE:  {np.mean(np.abs(y_true - y_pred)):.4f}%")
        print(f"   R2:   {1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2):.4f}")
        bias = np.mean(y_pred - y_true)
        bias_type = 'overestimation' if bias > 0 else 'underestimation'
        print(f"   Bias: {bias:.4f}% ({bias_type})")
        
        print(f"\nAll results saved in: {output_dir}/")
        return True
        
    except Exception as e:
        print(f"Error in plot_predictions_vs_reference: {e}")
        import traceback
        traceback.print_exc()
        return False
