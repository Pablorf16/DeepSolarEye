"""
plot_results.py
===============
Visualización de la curva de aprendizaje de DeepSolarEye.

Lee el fichero ``training_log.csv`` generado durante el entrenamiento y
produce una gráfica MSE de train vs. val con anotación del mejor modelo.

Uso::

    python src/plot_results.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve():
    """
    Lee ``training_log.csv`` y guarda una gráfica de la curva de aprendizaje.

    La gráfica muestra la evolución del MSE de entrenamiento y validación por
    época, e indica visualmente el punto donde se alcanzó la mejor validación
    (modelo guardado).
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_FILE = os.path.join(BASE_DIR, 'training_log.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] No se encuentra el archivo {LOG_FILE}.")
        return

    df = pd.read_csv(LOG_FILE)

    # Buscar la época con menor pérdida de validación (criterio de mejor modelo)
    best_epoch = df.loc[df['val_loss'].idxmin()]

    plt.figure(figsize=(10, 6))

    # Curvas de entrenamiento y validación
    plt.plot(
        df['epoch'], df['train_loss'],
        label='Train Loss (MSE)', color='#1f77b4',
        linewidth=2, marker='o', markersize=4,
    )
    plt.plot(
        df['epoch'], df['val_loss'],
        label='Val Loss (MSE)', color='#2ca02c',
        linewidth=2, marker='s', markersize=4,
    )

    # Marca roja en la mejor época
    plt.scatter(
        best_epoch['epoch'], best_epoch['val_loss'],
        color='red', s=100, zorder=5,
        label=f"Mejor Modelo (Época {int(best_epoch['epoch'])})",
    )
    plt.annotate(
        f"{best_epoch['val_loss']:.2f}",
        (best_epoch['epoch'], best_epoch['val_loss']),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center', fontsize=10, fontweight='bold', color='red',
    )

    plt.title('Curva de Aprendizaje - DeepSolarEye', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Épocas (Epochs)', fontsize=12)
    plt.ylabel('Error Cuadrático Medio (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10, shadow=True)
    plt.xticks(range(1, int(df['epoch'].max()) + 1))
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'learning_curve.png')
    plt.savefig(output_path, dpi=300)
    print(f"Gráfica generada y guardada en: {output_path}")


if __name__ == "__main__":
    plot_learning_curve()
