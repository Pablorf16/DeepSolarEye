

import torch








# Soiling categories (quartile-based stratification)
CATEGORY_BINS = [-1, 5, 15, 30, 60, 105]  # Power loss percentages: 0-100%
CATEGORY_LABELS = ['Limpio', 'Leve', 'Moderado', 'Alto', 'Crítico']

# Training hyperparameters
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
MAX_EPOCHS = 150
ES_PATIENCE = 15
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation metrics
OPTIMIZING_METRIC = 'rmse'  # Primary metric for model selection
DIAGNOSTIC_METRICS = ['mae', 'r2']  # Secondary metrics for analysis

# Data stratification (train/val/test split)
DATA_SPLIT = {
    'train': 0.60,
    'val': 0.20,
    'test': 0.20,
}
RANDOM_STATE = 42

# Data augmentation strategy (training phase)
AUGMENTATION_STRATEGY = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.5,
    'rotation_degrees': 180,
}

# ============================================================
# ARCHIVOS Y LOGGING
# ============================================================

# Formato de mensajes de logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Nivel de intensidad del logging
LOG_LEVEL = 'INFO'  # Niveles: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Nombre del archivo de historial de training
# Contiene: epoch, train_rmse, val_rmse, val_mae, val_r2, learning_rate,
#           rmse por categoría (v3.1+)
TRAINING_LOG_NAME = 'training_log_v4.0.csv'

# Nombre del archivo de checkpoint (para reanudar entrenamiento)
# Contiene: model_state_dict, optimizer_state_dict, best_val_rmse, epoch
CHECKPOINT_NAME = 'checkpoint_v4.0.pth'

# Nombre del archivo del mejor modelo encontrado
# Se guarda cuando: val_rmse < best_val_rmse
BEST_MODEL_NAME = 'best_model_v4.0.pth'

# Image input configuration
IMG_SIZE = 224
IMG_CHANNELS = 3

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# DIAGNÓSTICO DEL MODELO
# ============================================================

# Habilitar reporting de predicciones fuera [0, 100]
# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Training log and model checkpoint names
OUT_OF_BOUNDS_DIAGNOSTIC = True
OUT_OF_BOUNDS_MIN = 0
OUT_OF_BOUNDS_MAX = 100

# Environmental features
NUM_ENV_FEATURES = 1  # Irradiance

# Gradient clipping for training stability