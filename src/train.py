"""
train.py - Entrenamiento de DeepSolarEye v3.0

Implementación completa del plan de ejecución v3.0:
  1. Sin sigmoid en salida (regresión abierta)
  2. Stratified split documentado
  3. RMSE (optimizing) + R²,MAE (diagnostic)
  4. Early Stopping con PATIENCE=12
  5. ReduceLROnPlateau scheduler
  6. Oversampling + Data Augmentation (sin WeightedSampler)

Entrada: CSVs de entrenamiento, validación, test
Salida: Mejor modelo, checkpoint, training_log_v3.csv, gráficas
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
    LEARNING_RATE,
    MAX_EPOCHS,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
    SEED,
    TRAINING_LOG_NAME,
)
from src.dataset import SolarPanelDataset, get_transforms
from src.model import Net

# ============================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================

# Configurar logging para profesionalismo académico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Mostrar en consola
        # logging.FileHandler('training.log')  # Opcional: guardar en archivo
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN v3.0
# ============================================================

# Reproducibilidad: SEED fijo (importado de config.py)
# Aplicado a: torch, numpy, random, deterministic mode
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True



# Rutas dinámicas (usando pathlib para coherencia cross-platform)
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE_DIR / 'data' / 'processed' / 'train_dataset.csv'
VAL_CSV = BASE_DIR / 'data' / 'processed' / 'val_dataset.csv'
TEST_CSV = BASE_DIR / 'data' / 'processed' / 'test_dataset.csv'
IMG_DIR = BASE_DIR / 'data' / 'raw'
SAVE_DIR = BASE_DIR / 'saved_models'
LOG_FILE = BASE_DIR / TRAINING_LOG_NAME
CHECKPOINT_FILE = SAVE_DIR / CHECKPOINT_NAME

# ============================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    """
    Entrena el modelo durante una época.
    
    Calcula MSE promedio y retorna como RMSE para consistencia con validate().
    
    Args:
        model (nn.Module): Red neuronal en modo entrenamiento
        loader (DataLoader): Cargador de datos de entrenamiento
        criterion (nn.Module): Función de pérdida (MSELoss)
        optimizer (optim.Optimizer): Optimizador (Adam)
    
    Returns:
        float: RMSE de entrenamiento (raíz del MSE promedio)
    
    Features:
        - Barra de progreso con tqdm
        - Cálculo incremental de RMSE
        - Movimiento de tensores a dispositivo
    """
    model.train()
    total_mse = 0.0
    num_samples = 0
    
    # Barra de progreso durante época
    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        # Mover datos a dispositivo (GPU o CPU)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).float()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Backward pass
        loss = criterion(outputs.squeeze(dim=1), labels)  # MSE
        loss.backward()
        optimizer.step()
        
        # Acumular métrica
        total_mse += loss.item() * images.size(0)
        num_samples += images.size(0)
        loop.set_postfix(mse=loss.item())
    
    # Retornar RMSE (no MSE) para interpretabilidad
    mse = total_mse / num_samples
    rmse = np.sqrt(mse)
    
    return rmse


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, float, np.ndarray, np.ndarray, float]:
    """
    Evalúa el modelo en un conjunto de validación o test.
    
    Calcula múltiples métricas para evaluación exhaustiva:
    - RMSE: Métrica de optimización (penaliza errores grandes)
    - MAE, R²: Métricas de diagnóstico
    - Out-of-bounds: Indicador de problemas
    
    Args:
        model (nn.Module): Red neuronal en modo evaluación
        loader (DataLoader): Cargador de datos de validación/test
        criterion (nn.Module): Función de pérdida
    
    Returns:
        Tuple: (rmse, mae, r2, y_true, y_pred, out_of_bounds_pct)
            donde:
            - rmse (float): Raíz del error cuadrático medio
            - mae (float): Error absoluto medio
            - r2 (float): Coeficiente de determinación
            - y_true (np.ndarray): Labels verdaderos
            - y_pred (np.ndarray): Predicciones del modelo
            - out_of_bounds_pct (float): % de predicciones fuera [0, 100]
    """
    model.eval()
    total_mse = 0.0
    all_preds = []
    all_labels = []
    num_samples = 0
    
    # Evaluación sin cálculo de gradientes (mais rápido)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(dim=1), labels)
            
            # Acumular MSE
            total_mse += loss.item() * images.size(0)
            num_samples += images.size(0)
            
            # Guardar predicciones para métricas finales
            all_preds.extend(outputs.squeeze(dim=1).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Convertir listas a arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calcular métricas
    mse = total_mse / num_samples
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    # Diagnóstico: % de predicciones fuera de rango físico [0, 100]
    out_of_bounds = np.sum((all_preds < 0) | (all_preds > 100))
    out_of_bounds_pct = 100 * out_of_bounds / len(all_preds)
    
    return rmse, mae, r2, all_labels, all_preds, out_of_bounds_pct


def generate_final_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Genera reporte académico final con matriz de confusión.
    
    Discretiza predicciones continuas en categorías (Limpio, Leve, etc.)
    y genera matriz de confusión para análisis por categoría.
    
    Args:
        y_true (np.ndarray): Labels verdaderos en escala continua [0, 100]
        y_pred (np.ndarray): Predicciones del modelo [0, 100]
    """
    print("\n" + "=" * 60)
    print("📊 REPORTE FINAL DE VALIDACIÓN (TEST SET)")
    print("=" * 60)
    
    # Discretizar predicciones continuas en categorías
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
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=CATEGORY_LABELS)
    
    # Mostrar matriz de confusión
    print("\nMatriz de Confusión (Categorías de Suciedad):")
    print("      ", "  ".join(f"{l[:3]}" for l in CATEGORY_LABELS))
    for i, label in enumerate(CATEGORY_LABELS):
        print(f"{label[:3]}: ", "  ".join(f"{c:3d}" for c in cm[i]))
    
    # Calcular y mostrar precisión por categoría
    print("\nPrecisión por Categoría:")
    for i, label in enumerate(CATEGORY_LABELS):
        total = cm[i].sum()
        correct = cm[i, i] if total > 0 else 0
        acc = 100 * correct / total if total > 0 else 0
        print(f"  {label:12s}: {acc:6.2f}% ({correct}/{total})")
    
    print("=" * 60)


def main() -> None:
    """
    Función principal: orquesta todo el pipeline de entrenamiento.
    
    Flujo:
    1. Carga datasets (train oversampleado, val/test originales)
    2. Inicializa modelo, optimizer, scheduler
    3. Loop de entrenamiento con early stopping
    4. Evaluación final en test set
    5. Generación de gráficas
    """
    
    # ============================================================
    # INICIALIZACIÓN Y CONFIGURACIÓN
    # ============================================================
    
    print(f"\n{'='*60}")
    print("🚀 INICIANDO ENTRENAMIENTO DeepSolarEye v3.0")
    print("="*60)
    print(f"Dispositivo:        {DEVICE}")
    print(f"SEED:               {SEED}")
    print(f"Learning Rate:      {LEARNING_RATE}")
    print(f"Batch Size:         {BATCH_SIZE}")
    print(f"ES Patience:        {ES_PATIENCE}")
    print(f"Scheduler Patience: {SCHEDULER_PATIENCE}")
    print("="*60 + "\n")
    
    # Crear directorios de salida
    os.makedirs(str(SAVE_DIR), exist_ok=True)
    
    # ============================================================
    # 1. CARGA DE DATASETS
    # ============================================================
    
    logger.info("Cargando datasets...")
    try:
        # Train: oversampleado (filas duplicadas según categoría)
        train_ds = SolarPanelDataset(
            str(TRAIN_CSV),
            str(IMG_DIR),
            transform=get_transforms('train'),
            )
        
        # Val: original (sin oversample)
        val_ds = SolarPanelDataset(
            str(VAL_CSV),
            str(IMG_DIR),
            transform=get_transforms('test'),
            verbose=False
        )
        
        # Test: original (sin oversample)
        test_ds = SolarPanelDataset(
            str(TEST_CSV),
            str(IMG_DIR),
            transform=get_transforms('test'),
            verbose=False
        )
        
        print(f"   ✅ Train (Oversampleado): {len(train_ds)} muestras")
        print(f"   ✅ Val (Original):        {len(val_ds)} muestras")
        print(f"   ✅ Test (Original):       {len(test_ds)} muestras")
        
    except Exception as e:
        logger.error(f"Error cargando datasets: {e}")
        raise
    
    # Crear DataLoaders (sin WeightedRandomSampler, usamos shuffle normal)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # ============================================================
    # 2. INICIALIZACIÓN: MODELO, OPTIMIZER, SCHEDULER
    # ============================================================
    
    logger.info("Inicializando modelo y optimizador...")
    model = Net().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ReduceLROnPlateau: reduce LR cuando val_rmse no mejora
    # Nota: verbose=True está deprecated en PyTorch 2.0+, se utiliza logging en su lugar
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',           # Minimizar RMSE
        factor=SCHEDULER_FACTOR,  # LR *= SCHEDULER_FACTOR
        patience=SCHEDULER_PATIENCE,
    )

    print(f"   ✅ Modelo en {DEVICE}")
    print(f"   ✅ Optimizer: Adam(lr={LEARNING_RATE})")
    print(f"   ✅ Scheduler: ReduceLROnPlateau(patience={SCHEDULER_PATIENCE})")
    
    # ============================================================
    # 3. VARIABLES DE CONTROL DEL ENTRENAMIENTO
    # ============================================================
    
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    history = []
    start_epoch = 0
    
    # Intentar cargar checkpoint si existe (para reanudar)
    if CHECKPOINT_FILE.exists():
        logger.info("Encontrado checkpoint anterior, reanudando...")
        checkpoint = torch.load(str(CHECKPOINT_FILE), map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Restaurar scheduler
        start_epoch = checkpoint['epoch'] + 1
        best_val_rmse = checkpoint['best_val_rmse']
        epochs_no_improve = checkpoint['epochs_no_improve']
        
        # Cargar historial previo
        if LOG_FILE.exists():
            history = pd.read_csv(str(LOG_FILE)).to_dict('records')
        
        print(f"   ✅ Reanudando desde época {start_epoch + 1}")
        print(f"   ✅ Best val RMSE: {best_val_rmse:.4f}")
    else:
        logger.info("Empezando entrenamiento desde cero")
    
    # ============================================================
    # 4. LOOP DE ENTRENAMIENTO
    # ============================================================
    
    print(f"\n{'='*60}")
    print("INICIO DEL ENTRENAMIENTO")
    print("="*60 + "\n")
    
    try:
        for epoch in range(start_epoch, MAX_EPOCHS):
            print(f"[ Época {epoch+1}/{MAX_EPOCHS} ]")
            
            # Entrenar una época
            train_rmse = train_one_epoch(model, train_loader, criterion, optimizer)
            
            # Validar
            val_rmse, val_mae, val_r2, _, _, val_out_of_bounds = validate(
                model, val_loader, criterion
            )
            
            # Reportar métricas
            print(f"   📉 Train RMSE: {train_rmse:.4f}%")
            print(f"   🎯 Val RMSE:   {val_rmse:.4f}% (Optimizing Metric)")
            print(f"   📊 Val MAE:    {val_mae:.4f}%  | R²: {val_r2:.4f}")
            print(f"   🔍 Out-of-bounds: {val_out_of_bounds:.2f}%")
            
            # Learning rate scheduler step
            scheduler.step(val_rmse)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   📈 Learning Rate: {current_lr:.6f}")
            
            # Guardar historial en CSV
            history.append({
                'epoch': epoch + 1,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'val_out_of_bounds': val_out_of_bounds,
                'learning_rate': current_lr
            })
            pd.DataFrame(history).to_csv(str(LOG_FILE), index=False)
            
            # Early Stopping: Mejor modelo encontrado?
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_no_improve = 0
                torch.save(model.state_dict(), str(SAVE_DIR / BEST_MODEL_NAME))
                logger.info(f"¡Mejor modelo encontrado! RMSE = {best_val_rmse:.4f}")
                print(f"   ✅ ¡Mejor modelo! RMSE = {best_val_rmse:.4f}")
            else:
                epochs_no_improve += 1
                print(f"   ⏳ Sin mejora: {epochs_no_improve}/{ES_PATIENCE}")
            
            # Guardar checkpoint (para reanudar entrenamiento si se interrumpe)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # ReduceLROnPlateau state
                'best_val_rmse': best_val_rmse,
                'epochs_no_improve': epochs_no_improve
            }, str(CHECKPOINT_FILE))
            
            # Early Stopping: ¿Alcanzamos la paciencia?
            if epochs_no_improve >= ES_PATIENCE:
                logger.info(
                    f"Early stopping activado después de "
                    f"{epochs_no_improve} épocas sin mejora"
                )
                print(
                    f"\n🛑 EARLY STOPPING ACTIVADO "
                    f"(después de {epochs_no_improve} épocas sin mejora)"
                )
                break
            
            print()  # Línea en blanco entre épocas
    
    except KeyboardInterrupt:
        logger.warning("Entrenamiento interrumpido por usuario. Checkpoint guardado.")
        print("\n⚠️ Entrenamiento interrumpido por usuario. Checkpoint guardado.")
        return
    
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        print(f"\n❌ Error durante entrenamiento: {e}")
        traceback.print_exc()
        raise
    
    # ============================================================
    # 5. EVALUACIÓN FINAL EN TEST SET
    # ============================================================
    
    print(f"\n{'='*60}")
    print("EVALUACIÓN FINAL EN TEST SET")
    print("="*60 + "\n")
    
    # Cargar mejor modelo para evaluación final
    logger.info("Cargando mejor modelo guardado...")
    print("Cargando mejor modelo guardado...")
    model.load_state_dict(
        torch.load(str(SAVE_DIR / BEST_MODEL_NAME), map_location=DEVICE)
    )
    
    # Evaluar en test set
    test_rmse, test_mae, test_r2, y_true, y_pred, test_out_of_bounds = validate(
        model, test_loader, criterion
    )
    
    # Reportar resultados finales
    print(f"\n🎯 RESULTADOS FINALES TEST SET:")
    print(f"   RMSE: {test_rmse:.4f}%  (Métrica de Optimización)")
    print(f"   MAE:  {test_mae:.4f}%  (Diagnóstico)")
    print(f"   R²:   {test_r2:.4f}     (Diagnóstico)")
    print(f"   Out-of-bounds: {test_out_of_bounds:.2f}%")
    logger.info(
        f"Test Results - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}"
    )
    
    # Generar reporte detallado con matriz de confusión
    generate_final_report(y_true, y_pred)
    
    print(f"\n✅ Entrenamiento completado exitosamente")
    print(f"   Mejor modelo:        {SAVE_DIR / BEST_MODEL_NAME}")
    print(f"   Log entrenamiento:   {LOG_FILE}")
    logger.info("Entrenamiento completado exitosamente")
    
    # ============================================================
    # 6. GENERACIÓN DE GRÁFICAS
    # ============================================================
    
    print(f"\n📊 Generando gráficas de entrenamiento...")
    try:
        from src.plot_results import plot_training_curves_v3
        
        plot_training_curves_v3(str(LOG_FILE), str(SAVE_DIR))
        logger.info("Gráficas generadas con éxito")
        print("✅ Gráficas generadas con éxito")
    except Exception as e:
        logger.warning(f"No se pudieron generar gráficas: {e}")
        print(f"⚠️ No se pudieron generar gráficas: {e}")


if __name__ == "__main__":
    main()












