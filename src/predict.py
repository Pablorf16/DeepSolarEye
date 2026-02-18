import os
import torch
import random
from PIL import Image
from model import Net
from dataset import get_transforms

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'model_epoch_1.pth') # Usamos el que ya tienes
IMG_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_single_image():
    print("--- INICIANDO SISTEMA DE DIAGNÓSTICO DE PANELES ---")
    
    # 1. Cargar la Estructura del Modelo
    print(f"Cargando arquitectura neuronal...")
    model = Net()
    
    # 2. Cargar los 'Recuerdos' (Pesos entrenados)
    if os.path.exists(MODEL_PATH):
        print(f"Cargando pesos desde: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"ERROR: No encuentro el archivo {MODEL_PATH}")
        return

    # Ponemos el modelo en modo evaluación (NO entrenamiento)
    model.to(DEVICE)
    model.eval() 

    # 3. Seleccionar una imagen al azar para probar
    # (En una app real, aquí el usuario subiría su foto)
    all_images = os.listdir(IMG_DIR)
    random_image = random.choice(all_images)
    img_path = os.path.join(IMG_DIR, random_image)
    
    print(f"Analizando imagen: {random_image}")

    # 4. Preprocesar la imagen (Igual que en el entrenamiento)
    transform = get_transforms('test')
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0) # Añadimos dimensión de lote (Batch=1)
    input_tensor = input_tensor.to(DEVICE)

    # 5. Predicción (Inferencia)
    with torch.no_grad(): # No calculamos gradientes (ahorra memoria)
        prediction = model(input_tensor)
        percentage_loss = prediction.item()

    # 6. Informe de Resultados
    print("\n" + "="*30)
    print("      RESULTADO DEL ANÁLISIS")
    print("="*30)
    print(f"Panel Solar: {random_image}")
    # Acotamos entre 0 y 100 por estética (aunque el modelo puede pasarse)
    loss_visual = max(0, min(100, percentage_loss))
    print(f"Pérdida de Potencia Estimada: {loss_visual:.2f}%")
    
    # Interpretación básica
    if loss_visual < 10:
        print("Estado: LIMPIO (Eficiencia Óptima)")
    elif loss_visual < 30:
        print("Estado: SUCIEDAD LEVE (Monitorizar)")
    else:
        print("Estado: CRÍTICO (Requiere Limpieza Inmediata)")
    print("="*30 + "\n")

if __name__ == "__main__":
    predict_single_image()