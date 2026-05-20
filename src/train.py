"""
Pipeline de entrenamiento orquestado para DeepSolarEye v4.0.
Gestiona el bucle de entrenamiento, validación, early stopping tolerante y generación de reportes y gráficas del modelo convolucional
"""

import logging
import os
import random
import traceback
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE,
    BATCH_SIZE,
    BEST_MODEL_NAME,
    CATEGORY_BINS,
    CATEGORY_LABELS,
    CHECKPOINT_NAME,
    ES_PATIENCE,
    GRAD_CLIP_MAX_NORM,
    LEARNING_RATE,
    MAX_EPOCHS,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
    SEED,
    TRAINING_LOG_NAME,
)
from src.dataset import SolarPanelDataset, get_transforms
from src.model import Net
# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Reproducibilidad: establece semilla en todas las librerías para resultados consistentes
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# Rutas dinámicas para compatibilidad multiplataforma (Windows, Linux, macOS)
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE_DIR / 'data' / 'processed' / 'train_dataset.csv'
VAL_CSV = BASE_DIR / 'data' / 'processed' / 'val_dataset.csv'
TEST_CSV = BASE_DIR / 'data' / 'processed' / 'test_dataset.csv'
IMG_DIR = BASE_DIR / 'data' / 'raw'
SAVE_DIR = BASE_DIR / 'saved_models'
LOG_FILE = BASE_DIR / TRAINING_LOG_NAME
CHECKPOINT_FILE = SAVE_DIR / CHECKPOINT_NAME


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    """Entrena el modelo durante un episodio"""
    model.train()
    total_mse = 0.0
    num_samples = 0

    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels, env in loop:
        images, labels, env = (
            images.to(DEVICE),
            labels.to(DEVICE).float(),
            env.to(DEVICE)
        )

        optimizer.zero_grad()
        outputs = model(images, env)
        loss = criterion(outputs.squeeze(dim=1), labels)
        loss.backward()

        # Clipping de gradientes: evita gradientes explosivos durante backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        optimizer.step()

        total_mse += loss.item() * images.size(0)
        num_samples += images.size(0)
        loop.set_postfix(mse=loss.item())

    mse = total_mse / num_samples
    rmse = np.sqrt(mse)
    return rmse


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, dict]:
    """Evalúa el modelo en los conjuntos de validación o test.

    Returns:
        Tuple que contiene métricas de rendimiento (RMSE, MAE, R2), 
        vectores de predicción, porcentaje fuera de límites y RMSE por categoría.
    """
    model.eval()
    total_mse = 0.0
    all_preds = []
    all_labels = []
    num_samples = 0
    
    with torch.no_grad():
        for images, labels, env in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float()
            env = env.to(DEVICE)
            
            outputs = model(images, env)
            loss = criterion(outputs.squeeze(dim=1), labels)
            
            total_mse += loss.item() * images.size(0)
            num_samples += images.size(0)
            all_preds.extend(outputs.squeeze(dim=1).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mse = total_mse / num_samples
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    out_of_bounds = np.sum((all_preds < 0) | (all_preds > 100))
    out_of_bounds_pct = 100 * out_of_bounds / len(all_preds)
    
    # Calcula RMSE por categoría para análisis diagnóstico (Q1, Q2, Q3, Q4)
    rmse_by_cat = {}
    true_cats = pd.cut(
        all_labels,
        bins=CATEGORY_BINS,
        labels=CATEGORY_LABELS,
        include_lowest=True
    )
    for cat in CATEGORY_LABELS:
        mask = (true_cats == cat)
        if mask.sum() > 0:
            cat_mse = np.mean((all_preds[mask] - all_labels[mask]) ** 2)
            rmse_by_cat[cat] = np.sqrt(cat_mse)
        else:
            rmse_by_cat[cat] = 0.0
    
    return rmse, mae, r2, all_labels, all_preds, out_of_bounds_pct, rmse_by_cat


def generate_final_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Imprime por consola la matriz de confusión basada en las categorías de suciedad."""
    print("\n" + "=" * 60)
    print("FINAL VALIDATION REPORT (TEST SET)")
    print("=" * 60)
    
    y_true_cat = pd.cut(
        y_true,
        bins=CATEGORY_BINS,
        labels=CATEGORY_LABELS,
        include_lowest=True
    )
    y_pred_cat = pd.cut(
        y_pred,
        bins=CATEGORY_BINS,
        labels=CATEGORY_LABELS,
        include_lowest=True
    )
    
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=CATEGORY_LABELS)
    
    print("\nConfusion Matrix (Soiling Categories):")
    print("      ", "  ".join(f"{l[:3]}" for l in CATEGORY_LABELS))
    for i, label in enumerate(CATEGORY_LABELS):
        print(f"{label[:3]}: ", "  ".join(f"{c:3d}" for c in cm[i]))
    
    print("\nAccuracy per Category:")
    for i, label in enumerate(CATEGORY_LABELS):
        total = cm[i].sum()
        correct = cm[i, i] if total > 0 else 0
        acc = 100 * correct / total if total > 0 else 0
        print(f"  {label:12s}: {acc:6.2f}% ({correct}/{total})")
    
    print("=" * 60)


def main() -> None:
    """Orquesta el flujo principal de entrenamiento y evaluación."""
    
    print(f"\n{'='*60}")
    print("Starting Training Pipeline")
    print("="*60)
    print(f"Device:         {DEVICE}")
    print(f"SEED:           {SEED}")
    print(f"Learning Rate:  {LEARNING_RATE}")
    print(f"Batch Size:     {BATCH_SIZE}")
    print(f"ES Patience:    {ES_PATIENCE}")
    print(f"MAX Epochs:     {MAX_EPOCHS}")
    print("="*60 + "\n")
    
    os.makedirs(str(SAVE_DIR), exist_ok=True)
    
    logger.info("Loading datasets...")
    try:
        # Entrenamiento: equilibrado por cuartiles dinámicos (25% por categoría, sin oversampling)
        train_ds = SolarPanelDataset(
            str(TRAIN_CSV),
            str(IMG_DIR),
            transform=get_transforms('train'),
            )
        
        # Validación: estratificado, sin modificaciones
        val_ds = SolarPanelDataset(
            str(VAL_CSV),
            str(IMG_DIR),
            transform=get_transforms('test'),
            verbose=False
        )
        
        # Test: estratificado, sin modificaciones
        test_ds = SolarPanelDataset(
            str(TEST_CSV),
            str(IMG_DIR),
            transform=get_transforms('test'),
            verbose=False
        )
        
        print(f"   Train (Quartiles): {len(train_ds)} samples")
        print(f"   Val: {len(val_ds)} samples")
        print(f"   Test: {len(test_ds)} samples")
        
    except Exception as e:
        logger.error("Error loading datasets: %s", e)
        raise
    
    # drop_last=True: evita batches de tamaño 1 que causan error en BatchNorm2d
    # de las capas convolucionales de la CNN
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    logger.info("Initializing model and optimizer...")
    model = Net().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    print(f"   Model on: {DEVICE}")
    print(f"   Optimizer: Adam(lr={LEARNING_RATE})")
    print(f"   Scheduler: ReduceLROnPlateau(patience={SCHEDULER_PATIENCE})")
    
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    history = []
    start_epoch = 0
    
    if CHECKPOINT_FILE.exists():
        logger.info("Resuming training from checkpoint...")
        try:
            checkpoint = torch.load(str(CHECKPOINT_FILE), map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_rmse = checkpoint['best_val_rmse']
            epochs_no_improve = checkpoint['epochs_no_improve']

            if LOG_FILE.exists():
                history = pd.read_csv(str(LOG_FILE)).to_dict('records')

            print(f"   Resuming from epoch {start_epoch + 1}")
            print(f"   Best val RMSE: {best_val_rmse:.4f}")

        except RuntimeError as e:
            logger.warning("Checkpoint incompatible: %s", e)
            logger.warning("Starting training from scratch...")
    else:
        logger.info("Starting training from scratch")
    
    print(f"\n{'='*60}")
    print("TRAINING STARTED")
    print("="*60 + "\n")
    
    try:
        for epoch in range(start_epoch, MAX_EPOCHS):
            print(f"[Epoch {epoch+1}/{MAX_EPOCHS}]")
            
            train_rmse = train_one_epoch(model, train_loader, criterion, optimizer)
            
            val_rmse, val_mae, val_r2, _, _, val_out_of_bounds, val_rmse_by_cat = validate(
                model, val_loader, criterion
            )
            
            print(f"   Train RMSE: {train_rmse:.4f}%")
            print(f"   Val RMSE:   {val_rmse:.4f}% (optimizing metric)")
            print(f"   Val MAE:    {val_mae:.4f}% | R²: {val_r2:.4f}")
            print(f"   Out-of-bounds: {val_out_of_bounds:.2f}%")
            
            # Métricas por categoría para diagnóstico (Q1_Limpio, Q2_Moderado, Q3_Alto, Q4_Crítico)
            rmse_cat_str = " | ".join(
                f"{cat[:3]}:{val_rmse_by_cat[cat]:.2f}"
                for cat in CATEGORY_LABELS
            )
            print(f"   RMSE/Category: {rmse_cat_str}")
            
            # Early Stopping TOLERANTE v4.0
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(val_rmse)
            current_lr = optimizer.param_groups[0]['lr']
            lr_just_reduced = (current_lr < lr_before)
            
            print(f"   Learning Rate: {current_lr:.6f}", end="")
            if lr_just_reduced:
                print(f" (reduced from {lr_before:.6e})")
            else:
                print()
            
            history_entry = {
                'epoch': epoch + 1,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'val_out_of_bounds': val_out_of_bounds,
                'learning_rate': current_lr
            }
            for cat in CATEGORY_LABELS:
                history_entry[f'rmse_{cat.lower()}'] = val_rmse_by_cat[cat]
            
            history.append(history_entry)
            pd.DataFrame(history).to_csv(str(LOG_FILE), index=False)
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improve = 0
                torch.save(model.state_dict(), str(SAVE_DIR / BEST_MODEL_NAME))
                print(f"   Best model found. RMSE: {best_val_rmse:.4f}")
                
            else:
                epochs_no_improve += 1
                
                # Mecanismo TOLERANTE v4.0: cuando LR se reduce, reduce penalizador 1 época
                # Permite 5-8 épocas de adaptación adicionales antes de detener
                if lr_just_reduced:
                    epochs_no_improve = max(0, epochs_no_improve - 1)
                    print(f"   No improvement: {epochs_no_improve}/{ES_PATIENCE} "
                          f"(LR recently reduced)")
                else:
                    print(f"   No improvement: {epochs_no_improve}/{ES_PATIENCE}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_rmse': best_val_rmse,
                'epochs_no_improve': epochs_no_improve
            }, str(CHECKPOINT_FILE))
            
            if epochs_no_improve >= ES_PATIENCE:
                print(
                    f"\n EARLY STOPPING ACTIVATED "
                    f"(no improvement for {epochs_no_improve} epochs)"
                )
                break
            
            print()  # Línea en blanco entre épocas
    
    except KeyboardInterrupt:
        print("\n  Training interrupted by user. Checkpoint saved.")
        return
    
    except Exception as e:
        print(f"\nError during training: {e}")
        traceback.print_exc()
        raise
    
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION")
    print("="*60 + "\n")
    
    print(" Loading best model...")
    model.load_state_dict(
        torch.load(str(SAVE_DIR / BEST_MODEL_NAME), map_location=DEVICE)
    )
    
    test_rmse, test_mae, test_r2, y_true, y_pred, test_out_of_bounds, test_rmse_by_cat = validate(
        model, test_loader, criterion
    )
    
    print(f"\nFINAL TEST RESULTS:")
    print(f"   RMSE: {test_rmse:.4f}% (optimizing metric)")
    print(f"   MAE:  {test_mae:.4f}% (diagnostic)")
    print(f"   R²:   {test_r2:.4f} (diagnostic)")
    print(f"   Out-of-bounds: {test_out_of_bounds:.2f}%")
    
    print(f"\nRMSE per Category (Test Set):")
    for cat in CATEGORY_LABELS:
        print(f"   {cat:12s}: {test_rmse_by_cat[cat]:.4f}%")
    logger.info(
        "Test Results - RMSE: %.4f, MAE: %.4f, R²: %.4f", test_rmse, test_mae, test_r2
    )
    
    generate_final_report(y_true, y_pred)
    print(f"\n Training completed successfully")
    print(f"   Model: {SAVE_DIR / BEST_MODEL_NAME}")
    print(f"   Log: {LOG_FILE}")
    
    print(f"\n Generating training visualizations...")
    try:
        from src.plot_results import plot_training_curves_v3, plot_predictions_vs_reference
        
        plot_training_curves_v3(str(LOG_FILE), str(SAVE_DIR))
        print(" Training curves generated")
        
        
        # Generar gráficas de predicción vs referencia
        print(f" Generating prediction vs reference analysis...")
        test_df = pd.read_csv(str(TEST_CSV))
        plot_predictions_vs_reference(y_true, y_pred, test_df, str(SAVE_DIR))
        print(" Prediction analysis completed")
        
        
    except Exception as e:
        print(f"  Error generating visualizations (see log): {e}")


if __name__ == "__main__":
    main()













