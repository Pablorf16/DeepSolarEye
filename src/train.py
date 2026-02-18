import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  # NUEVO: Para guardar el historial
from torch.utils.data import DataLoader
from tqdm import tqdm 

from dataset import SolarPanelDataset, get_transforms
from model import Net 

# --- CONFIGURACIÓN ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 5  # Intenta llegar al final esta vez si puedes

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'train_dataset.csv')
TEST_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'test_dataset.csv')
IMG_DIR = os.path.join(BASE_DIR, 'data', 'raw')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
LOG_FILE = os.path.join(BASE_DIR, 'training_log.csv') # NUEVO: Archivo de registro

def train_one_epoch(model, loader, criterion, optimizer):
    """Entrena una vuelta completa (Train)."""
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc="Entrenando", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE, dtype=torch.float32)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)

def evaluate(model, loader, criterion):
    """NUEVO: Evalúa el modelo en el set de Test (Sin aprender)."""
    model.eval() # Modo examen (congela dropout)
    running_loss = 0.0
    with torch.no_grad(): # No calculamos gradientes
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE, dtype=torch.float32)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
    return running_loss / len(loader)

def main():
    print(f"--- Iniciando Entrenamiento DeepSolarEye v2 (Con Gráficas) ---")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Cargar Datos
    print("--> Cargando Datasets...")
    train_ds = SolarPanelDataset(TRAIN_CSV, IMG_DIR, transform=get_transforms('train'))
    test_ds = SolarPanelDataset(TEST_CSV, IMG_DIR, transform=get_transforms('test')) # NUEVO

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False) # NUEVO

    # 2. Modelo
    model = Net().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Lista para guardar el historial
    history = []

    # 4. Bucle Principal
    for epoch in range(NUM_EPOCHS):
        print(f"\nÉpoca {epoch+1}/{NUM_EPOCHS}")
        
        # Entrenar
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # NUEVO: Validar (Ver cómo lo hace en el examen)
        test_loss = evaluate(model, test_loader, criterion)
        
        print(f"Resultados -> Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # Guardar datos en la lista
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss
        })

        # Guardar modelo
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'model_epoch_{epoch+1}.pth'))

    # 5. Guardar el CSV final para la gráfica
    df = pd.DataFrame(history)
    df.to_csv(LOG_FILE, index=False)
    print(f"\n--- Historial guardado en: {LOG_FILE} ---")

if __name__ == "__main__":
    main()