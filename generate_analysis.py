#!/usr/bin/env python3


import torch
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

from src.config import DEVICE, BATCH_SIZE
from src.model import Net
from src.dataset import SolarPanelDataset, get_transforms
from src.plot_results import plot_predictions_vs_reference

# Rutas
BASE_DIR = Path(__file__).parent
BEST_MODEL_PATH = BASE_DIR / 'saved_models' / 'best_model_v4.0.pth'
TEST_CSV = BASE_DIR / 'data' / 'processed' / 'test_dataset.csv'
IMG_DIR = BASE_DIR / 'data' / 'raw'  # CORRECCIÓN: dataset.py espera data/raw, el CSV contiene las subcarpetas
SAVE_DIR = BASE_DIR / 'reports' / 'figures'

print("\n" + "=" * 70)
print("GENERACION DE ANALISIS - DeepSolarEye v4.0 (sin reentrenamiento)")
print("=" * 70 + "\n")

# 1. Cargar modelo entrenado
print("[1/4] Cargando modelo entrenado...")
model = Net().to(DEVICE)
model.load_state_dict(torch.load(str(BEST_MODEL_PATH), map_location=DEVICE))
# PRUEBA 1: Comenta esta línea para probar si BatchNorm está corrupto
model.eval()
print(f"   OK - Modelo cargado: {BEST_MODEL_PATH.name}")
print("   PRUEBA 1: model.eval() ACTIVO - Si las predicciones son todas iguales, probablemente BatchNorm está corrupto")

# 2. Cargar test dataset
print("\n[2/4] Cargando test set...")
test_df = pd.read_csv(TEST_CSV)
test_dataset = SolarPanelDataset(
    csv_path=str(TEST_CSV),
    img_dir=str(IMG_DIR),
    transform=get_transforms('test'),
    verbose=False
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"   OK - Test set: {len(test_df)} muestras")

# 3. Hacer predicciones
print("\n[3/4] Realizando predicciones...")
y_true_list = []
y_pred_list = []

with torch.no_grad():
    for batch_idx, (images, labels, env) in enumerate(test_loader):
        # Mover tensores al dispositivo
        images = images.to(DEVICE)
        env = env.to(DEVICE)
        
        # Predicción SIN clamp: valores crudos del modelo para diagnóstico
        outputs = model(images, env)
        
        # Convertir a numpy manteniendo consistencia dimensional
        y_true_list.append(labels.cpu().numpy().flatten())
        y_pred_list.append(outputs.cpu().numpy().flatten())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"   Procesadas {(batch_idx + 1) * BATCH_SIZE}/{len(test_df)} muestras")

# Concatenar arrays
y_true = np.concatenate(y_true_list, axis=0)
y_pred = np.concatenate(y_pred_list, axis=0)
print(f"   OK - Predicciones completadas: {len(y_pred)} muestras")

# 4. Generar gráficas
print("\n[4/4] Generando gráficas de análisis...")
success = plot_predictions_vs_reference(y_true, y_pred, test_df, str(SAVE_DIR))

if success:
    print("\n" + "=" * 70)
    print("✓ ANALISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\nArchivos generados en: reports/figures/")
    print("  * scatter_pred_vs_ref.png")
    print("  * residuals_analysis.png")
    print("  * performance_by_category.png")
else:
    print("\nERROR en generacion de graficas")
