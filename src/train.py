"""
train.py
========
Bucle de entrenamiento para DeepSolarEye.

Entrena la red ``Net`` definida en ``model.py`` usando los CSVs generados
por ``data_prep.py``.  Incluye:
    - Semilla fija para reproducibilidad.
    - WeightedRandomSampler para balanceo de clases.
    - Seguimiento de MSE y MAE en train / val / test por época.
    - Early stopping basado en la pérdida de validación (sin data leakage).
    - Checkpointing para poder reanudar el entrenamiento.

Uso::

    python src/train.py

Requiere haber ejecutado previamente ``data_prep.py``.
"""

import os
import random
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import SolarPanelDataset, get_transforms
from model import Net

# ---------------------------------------------------------------------------
# Semilla global para reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# Hiperparámetros de entrenamiento
# ---------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
PATIENCE = 7       # Épocas sin mejora antes del early stopping

# ---------------------------------------------------------------------------
# Rutas del proyecto (relativas a este fichero)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'train_dataset.csv')
VAL_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'val_dataset.csv')
TEST_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'test_dataset.csv')
IMG_DIR = os.path.join(BASE_DIR, 'data', 'raw')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
LOG_FILE = os.path.join(BASE_DIR, 'training_log.csv')
CHECKPOINT_FILE = os.path.join(SAVE_DIR, 'checkpoint.pth')


def get_weighted_sampler(csv_path):
    """
    Crea un ``WeightedRandomSampler`` que balancea las clases de suciedad.

    Asigna a cada muestra un peso inversamente proporcional al número de
    muestras de su categoría, de modo que las clases minoritarias se
    muestreen con más frecuencia.

    Parameters
    ----------
    csv_path : str
        Ruta al CSV de entrenamiento (debe contener la columna
        ``dirt_category``).

    Returns
    -------
    WeightedRandomSampler
        Sampler listo para usar en ``DataLoader``.
    """
    df = pd.read_csv(csv_path)

    # Peso de cada clase = 1 / número de muestras de esa clase
    class_counts = df['dirt_category'].value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[cls] for cls in df['dirt_category']]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(model, loader, criterion_mse, criterion_mae, optimizer):
    """
    Realiza una época completa de entrenamiento.

    Parameters
    ----------
    model : Net
        Red neuronal en modo entrenamiento.
    loader : DataLoader
        DataLoader del conjunto de entrenamiento.
    criterion_mse : nn.MSELoss
        Función de pérdida MSE (usada para actualizar los pesos).
    criterion_mae : nn.L1Loss
        Función de pérdida MAE (solo para monitorización).
    optimizer : torch.optim.Optimizer
        Optimizador (Adam).

    Returns
    -------
    tuple[float, float]
        (mse_medio, mae_medio) sobre todos los batches de la época.
    """
    model.train()
    running_mse = 0.0
    running_mae = 0.0
    loop = tqdm(loader, desc="Entrenando", leave=False)

    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)

        # MSE se usa para actualizar los pesos; MAE solo se registra
        loss_mse = criterion_mse(outputs.squeeze(), labels)
        loss_mae = criterion_mae(outputs.squeeze(), labels)

        loss_mse.backward()
        optimizer.step()

        running_mse += loss_mse.item()
        running_mae += loss_mae.item()
        loop.set_postfix(mse=loss_mse.item(), mae=loss_mae.item())

    return running_mse / len(loader), running_mae / len(loader)


def evaluate(model, loader, criterion_mse, criterion_mae):
    """
    Evalúa el modelo sobre un conjunto de datos sin actualizar los pesos.

    Parameters
    ----------
    model : Net
        Red neuronal en modo evaluación.
    loader : DataLoader
        DataLoader del conjunto de validación o test.
    criterion_mse : nn.MSELoss
        Función de pérdida MSE.
    criterion_mae : nn.L1Loss
        Función de pérdida MAE.

    Returns
    -------
    tuple[float, float]
        (mse_medio, mae_medio) sobre todos los batches del loader.
    """
    model.eval()
    running_mse = 0.0
    running_mae = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.float32)
            outputs = model(images)

            running_mse += criterion_mse(outputs.squeeze(), labels).item()
            running_mae += criterion_mae(outputs.squeeze(), labels).item()

    return running_mse / len(loader), running_mae / len(loader)


def main():
    """
    Punto de entrada principal del entrenamiento.

    Carga los datasets, construye el modelo y ejecuta el bucle de
    entrenamiento con early stopping y checkpointing.
    """
    print("--- Iniciando Entrenamiento DeepSolarEye ---")
    print(f"-> Dispositivo: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Verificar que los CSVs existen antes de continuar
    for csv_path, csv_name in [(TRAIN_CSV, 'TRAIN'), (VAL_CSV, 'VAL'), (TEST_CSV, 'TEST')]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_name} CSV no encontrado: {csv_path}\n"
                "Ejecuta primero: python src/data_prep.py"
            )

    print("-> Cargando Datasets...")
    try:
        train_ds = SolarPanelDataset(TRAIN_CSV, IMG_DIR, transform=get_transforms('train'))
        val_ds = SolarPanelDataset(VAL_CSV, IMG_DIR, transform=get_transforms('test'))
        test_ds = SolarPanelDataset(TEST_CSV, IMG_DIR, transform=get_transforms('test'))
    except Exception as e:
        print(f"Error al cargar datasets: {e}")
        raise

    print("-> Calculando pesos para balanceo de clases...")
    train_sampler = get_weighted_sampler(TRAIN_CSV)

    # DataLoaders: train usa el sampler ponderado; val y test son secuenciales
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo, funciones de pérdida y optimizador
    model = Net().to(DEVICE)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Estado del entrenamiento (se sobreescribe si existe checkpoint)
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    # Reanudar desde checkpoint si existe
    if os.path.exists(CHECKPOINT_FILE):
        print("Checkpoint encontrado. Reanudando entrenamiento anterior...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']

        if os.path.exists(LOG_FILE):
            history = pd.read_csv(LOG_FILE).to_dict('records')
        print(f"-> Reanudando desde la época {start_epoch + 1}")
    else:
        print("-> Empezando entrenamiento desde cero.")

    # Bucle principal de entrenamiento
    try:
        for epoch in range(start_epoch, MAX_EPOCHS):
            print(f"\n[ Época {epoch + 1}/{MAX_EPOCHS} ]")

            train_mse, train_mae = train_one_epoch(
                model, train_loader, criterion_mse, criterion_mae, optimizer
            )
            val_mse, val_mae = evaluate(model, val_loader, criterion_mse, criterion_mae)
            test_mse, test_mae = evaluate(model, test_loader, criterion_mse, criterion_mae)

            print(f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Test MSE: {test_mse:.4f}")
            print(f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | Test MAE: {test_mae:.4f}")

            # Registrar métricas de la época en el historial
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_mse,
                'val_loss': val_mse,
                'test_loss': test_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'test_mae': test_mae,
            })
            pd.DataFrame(history).to_csv(LOG_FILE, index=False)

            # Early stopping basado en val_loss (evita data leakage con el test set)
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
                print(f"Mejor modelo guardado. (Val MSE: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"Sin mejora durante {epochs_no_improve} época(s).")

            # Guardar checkpoint para poder interrumpir y reanudar el entrenamiento
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve,
            }, CHECKPOINT_FILE)

            if epochs_no_improve >= PATIENCE:
                print(f"\nEARLY STOPPING: Convergencia alcanzada tras {PATIENCE} épocas sin mejora.")
                break

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario. Checkpoint guardado.")
    except Exception as e:
        print(f"\nError durante entrenamiento: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
