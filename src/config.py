import torch

# Dispositivo de cálculo (GPU si disponible, sino CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ESTRATIFICACIÓN DE DATOS Y CATEGORIZACIÓN
# Categorías de severidad de ensuciamiento (estratificación basada en cuartiles)
CATEGORY_BINS = [-1, 5, 15, 30, 60, 105]
CATEGORY_LABELS = ['Limpio', 'Leve', 'Moderado', 'Alto', 'Crítico']

# División train/validación/test (asegura estratificación en todos los splits)
DATA_SPLIT = {
    'train': 0.60,
    'val': 0.20,
    'test': 0.20,
}
RANDOM_STATE = 42


# HIPERPARÁMETROS DE ENTRENAMIENTO

# Semilla de reproducibilidad para PyTorch y NumPy
SEED = 42

# Tamaño de lote (batch size) para entrenamiento, validación y prueba
BATCH_SIZE = 32

# Tasa de aprendizaje inicial para el optimizador Adam
LEARNING_RATE = 0.0001

# Número máximo de episodios de entrenamiento
MAX_EPOCHS = 150

# Paciencia para Early Stopping (detener si sin mejora en N episodios)
ES_PATIENCE = 15

# Paciencia para ReduceLROnPlateau (reducir tasa si sin mejora en N episodios)
SCHEDULER_PATIENCE = 7

# Factor de reducción para la tasa de aprendizaje (LR * 0.5)
SCHEDULER_FACTOR = 0.5

# Valor máximo para clipping de gradientes (estabilidad numérica)
GRAD_CLIP_MAX_NORM = 1.0


# MÉTRICAS DE EVALUACIÓN

# Métrica primaria para selección de modelo durante entrenamiento
# Usada para Early Stopping y ajuste de tasa de aprendizaje
OPTIMIZING_METRIC = 'rmse'

# Métricas secundarias para análisis diagnóstico e informes
DIAGNOSTIC_METRICS = ['mae', 'r2']



# PROCESAMIENTO DE IMÁGENES Y NORMALIZACIÓN


# Tamaño estándar de entrada de imagen para el modelo CNN
IMG_SIZE = 224

# Número de canales de color (RGB)
IMG_CHANNELS = 3

# Parámetros de normalización de ImageNet (alineación con pre-entrenamiento)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]




# CARACTERÍSTICAS AMBIENTALES

# Número de características ambientales auxiliares (irradiancia)
NUM_ENV_FEATURES = 1



# VALIDACIÓN DE SALIDA DEL MODELO

# Habilitar reporte diagnóstico de predicciones fuera de rango
OUT_OF_BOUNDS_DIAGNOSTIC = True

# Rango válido de salida para porcentaje de pérdida de potencia
OUT_OF_BOUNDS_MIN = 0
OUT_OF_BOUNDS_MAX = 100



# LOGGING Y PUNTOS DE CONTROL DEL MODELO

# Formato de cadena para todos los mensajes de salida
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = 'INFO'

# Nombre del archivo de historial de entrenamiento (formato CSV)
# Columnas: epoch, train_rmse, val_rmse, val_mae, val_r2, learning_rate
TRAINING_LOG_NAME = 'training_log_v4.0.csv'

# Nombre de archivo de punto de control (para reanudar entrenamiento)
# Contiene: model_state_dict, optimizer_state_dict, best_val_rmse, epoch
CHECKPOINT_NAME = 'checkpoint_v4.0.pth'

# Nombre del archivo del mejor modelo
BEST_MODEL_NAME = 'best_model_v4.0.pth'