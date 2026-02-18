import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, 'training_log.csv')
OUTPUT_IMG = os.path.join(BASE_DIR, 'grafica_entrenamiento.png')

def plot_learning_curves():
    print(f"--- GENERANDO GRÁFICA DE APRENDIZAJE ---")
    
    # 1. Cargar el historial
    if not os.path.exists(LOG_FILE):
        print(f"ERROR: No encuentro el archivo {LOG_FILE}")
        print("¿Ha terminado el entrenamiento?")
        return

    data = pd.read_csv(LOG_FILE)
    
    # 2. Configurar el estilo del gráfico (Estilo Académico)
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot') # Estilo bonito y limpio
    
    # 3. Dibujar las líneas
    # Línea Azul: Error en Entrenamiento (Lo que memoriza)
    plt.plot(data['epoch'], data['train_loss'], 
             label='Train Loss (Entrenamiento)', 
             color='tab:blue', marker='o', linewidth=2)
    
    # Línea Roja: Error en Test (La realidad)
    plt.plot(data['epoch'], data['test_loss'], 
             label='Test Loss (Validación)', 
             color='tab:red', marker='x', linestyle='--', linewidth=2)
    
    # 4. Etiquetas y Títulos
    plt.title('Evolución del Error (MSE) - DeepSolarEye', fontsize=14)
    plt.xlabel('Épocas (Vueltas)', fontsize=12)
    plt.ylabel('Pérdida (MSE Loss)', fontsize=12)
    
    # Forzar que el eje X muestre enteros (1, 2, 3, 4, 5)
    plt.xticks(data['epoch'])
    
    plt.legend()
    plt.grid(True)
    
    # 5. Guardar
    plt.savefig(OUTPUT_IMG, dpi=300) # 300 dpi = Calidad de Impresión
    print(f"✅ Gráfica guardada en: {OUTPUT_IMG}")
    print("¡Ya puedes abrirla y enviársela a tu tutor!")
    
    # Mostrar en ventana (opcional)
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()