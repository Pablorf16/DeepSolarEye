

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Reproducibility: set seed across all libraries
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# Dynamic paths for cross-platform compatibility
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
    """Train model for one epoch. Returns training RMSE."""
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

        # Gradient clipping prevents exploding gradients
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
    """Evaluate model on validation/test set with comprehensive metrics.
    
    Returns: (rmse, mae, r2, y_true, y_pred, out_of_bounds_pct, rmse_by_cat)
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
    
    # Compute RMSE per category for diagnostic analysis
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
    """Generate confusion matrix report with categorical discretization."""
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
    """Main training orchestration pipeline."""
    
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
        # Train: equilibrado por cuartiles (25% por categoría, NO oversample)
        train_ds = SolarPanelDataset(
            str(TRAIN_CSV),
            str(IMG_DIR),
            transform=get_transforms('train'),
            )
        
        # Val: original (estratificado, sin modificaciones)
        val_ds = SolarPanelDataset(
            str(VAL_CSV),
            str(IMG_DIR),
            transform=get_transforms('test'),
            verbose=False
        )
        
        # Test: original (estratificado, sin modificaciones)
        test_ds = SolarPanelDataset(
            str(TEST_CSV),
            str(IMG_DIR),
            transform=get_transforms('test'),
            verbose=False
        )
        
        print(f"   Train (Cuartiles): {len(train_ds)} samples")
        print(f"   Val: {len(val_ds)} samples")
        print(f"   Test: {len(test_ds)} samples")
        
    except Exception as e:
        logger.error(f"Error cargando datasets: {e}")
        raise
    
    # Crear DataLoaders (sin WeightedRandomSampler, usamos shuffle normal)
    # drop_last=True: Evita batches de tamaño 1 que causan error en
    # BatchNorm2d de los Analysis Units de la CNN
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
            logger.warning(f"Checkpoint incompatible: {e}")
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
            
            
            rmse_cat_str = " | ".join(
                f"{cat[:3]}:{val_rmse_by_cat[cat]:.2f}"
                for cat in CATEGORY_LABELS
            )
            print(f"   RMSE/Cat: {rmse_cat_str}")
            
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
                logger.info(f"Best model found. RMSE: {best_val_rmse:.4f}")
                print(f"   Best model found. RMSE: {best_val_rmse:.4f}")
            else:
                epochs_no_improve += 1
                
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
                logger.info(
                    f"Early stopping triggered after {epochs_no_improve} epochs without improvement"
                )
                print(
                    f"\nEARLY STOPPING TRIGGERED "
                    f"(no improvement for {epochs_no_improve} epochs)"
                )
                break
            
            print()  # Línea en blanco entre épocas
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Checkpoint saved.")
        print("\nTraining interrupted by user. Checkpoint saved.")
        return
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        print(f"\nTraining error: {e}")
        traceback.print_exc()
        raise
    
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION")
    print("="*60 + "\n")
    
    logger.info("Loading best model...")
    print("Loading best model...")
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
        f"Test Results - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}"
    )
    
    generate_final_report(y_true, y_pred)
    
    print(f"\nTraining completed successfully")
    print(f"   Best model: {SAVE_DIR / BEST_MODEL_NAME}")
    print(f"   Training log: {LOG_FILE}")
    logger.info("Training completed successfully")
    # ============================================================
    
    print(f"\n📊 Generando gráficas de entrenamiento...")
    try:
        from src.plot_results import plot_training_curves_v3, plot_predictions_vs_reference
        
        plot_training_curves_v3(str(LOG_FILE), str(SAVE_DIR))
        logger.info("Gráficas de entrenamiento generadas con éxito")
        print("✅ Gráficas de entrenamiento generadas con éxito")
        
        # Generar gráficas de predicción vs referencia (Feedback tutor #3)
        print(f"\n📊 Generando análisis de predicción vs referencia...")
        test_df = pd.read_csv(str(TEST_CSV))
        plot_predictions_vs_reference(y_true, y_pred, test_df, str(SAVE_DIR))
        logger.info("Análisis de predicción vs referencia generado con éxito")
        print("✅ Análisis completado con éxito")
        
    except Exception as e:
        logger.warning(f"No se pudieron generar gráficas: {e}")
        print(f"⚠️ No se pudieron generar gráficas: {e}")


if __name__ == "__main__":
    main()













