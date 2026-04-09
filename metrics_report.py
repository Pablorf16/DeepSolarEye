#!/usr/bin/env python3
"""
Script para generar análisis detallado de métricas
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from src.config import DEVICE, BATCH_SIZE, CATEGORY_BINS, CATEGORY_LABELS
from src.model import Net
from src.dataset import SolarPanelDataset, get_transforms
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE_DIR = Path('.')
BEST_MODEL_PATH = BASE_DIR / 'saved_models' / 'best_model_v4.0.pth'
TEST_CSV = BASE_DIR / 'data' / 'processed' / 'test_dataset.csv'
IMG_DIR = BASE_DIR / 'data' / 'raw'

# Load model and data
print("Loading model...")
model = Net().to(DEVICE)
model.load_state_dict(torch.load(str(BEST_MODEL_PATH), map_location=DEVICE))
model.eval()

print("Loading dataset...")
test_dataset = SolarPanelDataset(
    csv_path=str(TEST_CSV),
    img_dir=str(IMG_DIR),
    transform=get_transforms('test'),
    verbose=False
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Get predictions
print("Generating predictions...")
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels, env in test_loader:
        images = images.to(DEVICE)
        env = env.to(DEVICE)
        outputs = model(images, env)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy().flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Análisis por categoría
y_category = pd.cut(y_true, bins=CATEGORY_BINS, labels=CATEGORY_LABELS, include_lowest=True)

print("\n" + "="*70)
print("MÉTRICAS DE DESEMPEÑO - DeepSolarEye v4.0 (TEST SET)")
print("="*70)

print(f"\n📊 MÉTRICAS GLOBALES:")
print(f"   Total muestras: {len(y_pred)}")
print(f"   RMSE: {rmse:.4f}%")
print(f"   MAE:  {mae:.4f}%")
print(f"   R²:   {r2:.4f}")

print(f"\n📈 ESTADÍSTICAS DE PREDICCIÓN:")
print(f"   Pred - mean: {np.mean(y_pred):.2f}%, std: {np.std(y_pred):.2f}%, min: {np.min(y_pred):.2f}%, max: {np.max(y_pred):.2f}%")
print(f"   True - mean: {np.mean(y_true):.2f}%, std: {np.std(y_true):.2f}%, min: {np.min(y_true):.2f}%, max: {np.max(y_true):.2f}%")

print(f"\n📊 DESEMPEÑO POR CATEGORÍA:")
for label in CATEGORY_LABELS:
    mask = (y_category == label)
    if mask.sum() > 0:
        cat_mae = mean_absolute_error(y_true[mask], y_pred[mask])
        cat_r2 = r2_score(y_true[mask], y_pred[mask])
        print(f"   {label:12s}: MAE={cat_mae:.2f}%, R²={cat_r2:.4f} (n={mask.sum()})")

print(f"\n✅ EVALUACIÓN:")
if r2 > 0.90:
    print(f"   ✓ Correlación EXCELENTE (R² = {r2:.4f} > 0.90)")
else:
    print(f"   ⚠ Correlación BUENA (R² = {r2:.4f})")

if mae < 10:
    print(f"   ✓ Error muy aceptable (MAE = {mae:.4f}% < 10%)")
    
print("\n" + "="*70)
