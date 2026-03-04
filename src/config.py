"""
config.py - Configuración centralizada para DeepSolarEye v3.0

Single source of truth para todos los parámetros del proyecto.
Esto facilita:
  1. Evitar duplicación: Un cambio = un lugar
  2. Sincronización: Todos los módulos leen los mismos valores
  3. Reproducibilidad: SEED centralizado y controlado
"""

import torch

# ============================================================
# CATEGORÍAS DE SUCIEDAD (Bins Estratificados)
# ============================================================

# Límites de bins para discretizar power_loss continuo en clases
# Corresponden a los cinco níveis de estado del panel
CATEGORY_BINS = [-1, 5, 15, 30, 60, 105]  # Escala 0-100%

# Etiquetas de categorías (nombres descriptivos)
CATEGORY_LABELS = ['Limpio', 'Leve', 'Moderado', 'Alto', 'Crítico']

"""
JUSTIFICACIÓN ACADÉMICA DE LÍMITES:
─────────────────────────────────────────

Categoría 1: Limpio (0-5%)
  Interpretación: Panel prácticamente nuevo, excelente condición
  Criterio: Industria solar (IEC 61724-1) considera <5% "excelente"
  Aplicación: Paneles recién limpios o sin suciedad acumulada

Categoría 2: Leve (5-15%)
  Interpretación: Suciedad ligera visible, operación levemente afectada
  Criterio: ~15% es el umbral donde propietarios típicamente intervienen
  Aplicación: Requiere limpieza en próximas semanas

Categoría 3: Moderado (15-30%)
  Interpretación: Suciedad media, impacto operacional claro
  Criterio: Degradación visible pero sin daño estructural
  Aplicación: Patrón de soiling típico sin intervención

Categoría 4: Alto (30-60%)
  Interpretación: Suciedad severa, degradación significativa
  Criterio: Pérdida de ingresos notable
  Aplicación: Requiere limpieza urgente

Categoría 5: Crítico (60-100%)
  Interpretación: Pérdida catastrófica, panel inutilizable
  Criterio: ~Casi sin producción, posible daño físico
  Aplicación: Situación urgente o desperfecto estructural
"""

# ============================================================
# HYPERPARÁMETROS DE ENTRENAMIENTO v3.0
# ============================================================

# Reproducibilidad: SEED FIJO para resultados determinísticos
# Afecta a: torch, numpy, random, sklearn.train_test_split
SEED = 42

# ----- Arquitectura y optimización -----------

# Tamaño de batch: balance entre memoria GPU y estabilidad del gradiente
# Ración recomendada:
#   - Datasets pequeños (<10k): 32-64
#   - Datasets medianos (10k-100k): 32-128
#   - Datasets grandes (>100k): 128-256
# Con oversampling en train, tenemos ~10k muestras → 32 es apropiado
# Verificar en primeras 5 épocas que gradientes sean estables
BATCH_SIZE = 32

# Learning rate inicial para optimizador Adam
# Rango típico para regresión: [0.0001, 0.01]
# 0.001 = valor conservador, seguro en regresión
# ReduceLROnPlateau reducirá aún más si val_rmse no mejora
# Validación: Monitorear train/val loss primeras 10 épocas
# - Si loss baja bien: 0.001 es correcto
# - Si loss sube: reducir a 0.0005 o 0.0001
# - Si baja demasiado lento: aumentar a 0.002-0.005
LEARNING_RATE = 0.001

# Máximo de épocas permitidas
# Con early stopping (PATIENCE=12), raramente se alcanza MAX_EPOCHS
# Típicamente: convergencia ~20-30 épocas con oversampling
# Aumentar a 100-200 recomendado si early stopping no se activa
MAX_EPOCHS = 50

# ----- Early Stopping y Scheduler -----------

# Early Stopping: Paciencia (épocas sin mejora antes de detener)
# AUMENTADO a 12 porque:
#   1. Oversampling expande dataset → más épocas para converger
#   2. Data augmentation añade variabilidad → gradientes más ruidosos
#   3. ReduceLROnPlateau reduce LR gradualmente → permite recuperación
# Configuración anterior (v2): PATIENCE=7
ES_PATIENCE = 12

# ReduceLROnPlateau: Paciencia antes de reducir learning rate
# Debe ser MENOR que ES_PATIENCE para que scheduler actúe primero
# Patrón: Scheduler reduce LR → modelo se recupera → ES lo ve
# Si val_rmse no mejora durante 5 épocas → LR *= 0.1
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.1  # Factor de reducción: LR_nuevo = LR_actual * factor

# Dispositivo de computación (GPU si disponible, CPU en caso contrario)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# ESTRATEGIA DE MÉTRICAS v3.0
# ============================================================

# Métrica principal para optimización
# Penaliza errores grandes (efecto cuadrático)
# Utilizada en: Early Stopping, ReduceLROnPlateau
OPTIMIZING_METRIC = 'rmse'

# Métricas de diagnóstico (no se usan para decisiones, solo análisis)
# MAE: Error absoluto promedio, fácil de interpretar
# R²: Varianza explicada, indica capacidad predictiva general
DIAGNOSTIC_METRICS = ['mae', 'r2']

"""
ESTRATEGIA DUAL DE MÉTRICAS (Plan v3.0):
──────────────────────────────────────────

RMSE (ROOT MEAN SQUARED ERROR) - OPTIMIZING METRIC
└─ Definición: √(Σ(y_true - y_pred)²) / N
└─ Rango: [0, ∞)
└─ Interpretación: "magnitud promedio de errores (penaliza grandes errores)"
└─ Uso: Decide cuál es el mejor modelo para guardar
└─ Por qué: Prioriza minimizar errores grandes (importa para regresión)

MAE (MEAN ABSOLUTE ERROR) - DIAGNOSTIC METRIC
└─ Definición: Σ|y_true - y_pred| / N
└─ Rango: [0, ∞)
└─ Interpretación: "en promedio erras X puntos de porcentaje"
└─ Uso: Entender magnitud real de errores de forma legible
└─ Valor típico: MAE ≈ 0.5 × RMSE (MAE < RMSE siempre)

R² (COEFICIENTE DE DETERMINACIÓN) - DIAGNOSTIC METRIC
└─ Definición: 1 - (SS_res / SS_tot)
└─ Rango: (-∞, 1], típicamente [0, 1]
└─ Interpretación: "% de varianza explicada por el modelo"
└─ Valores de referencia:
     R² ≥ 0.9: Excelente (rara vez en soiling)
     R² ≥ 0.8: Bueno (objetivo alcanzable)
     R² ≥ 0.7: Aceptable (baseline)
     R² < 0.5: Modelo no funciona
└─ Uso: Evaluar capacidad predictiva general
└─ R² negativo: Modelo peor que predicción constante → revisar

OUT-OF-BOUNDS (%) - DIAGNOSTIC METRIC
└─ Definición: % de predicciones fuera rango [0, 100]
└─ Rango: [0%, 100%]
└─ Interpretación: "modelo predice valores físicamente imposibles"
└─ Uso: Detectar problemas de aprendizaje (sin sigmoid = diagnóstico)
└─ Valor esperado: 0-5% es normal
└─ Valor crítico: >20% sugiere ajustar learning rate o arquitectura
"""

# ============================================================
# DIVISIÓN DE DATOS (Train / Validation / Test)
# ============================================================

# Proporción de datos en cada split (antes de oversampling)
# La suma debe ser 1.0
# Implementación: train_test_split estratificado en data_prep.py
DATA_SPLIT = {
    'train': 0.60,  # 60% para entrenamiento (será oversampleado)
    'val': 0.20,    # 20% para validación (original, sin oversample)
    'test': 0.20,   # 20% para pruebas finales (original, sin oversample)
}

# RANDOM_STATE: Especifica semilla para sklearn train_test_split
# Mismo valor en data_prep.py y train.py garantiza reproducibilidad
RANDOM_STATE = 42

# ============================================================
# ESTRATEGIA DE AUMENTACIÓN DE DATOS (Data Augmentation)
# ============================================================

# Diccionario de parámetros de augmentación
# Documentación detallada: ver dataset.py :: get_transforms()
AUGMENTATION_STRATEGY = {
    'horizontal_flip': 0.5,      # Probabilidad: 50% de volteo horizontal
    'vertical_flip': 0.5,        # Probabilidad: 50% de volteo vertical
    'rotation_degrees': 180,     # Rango de rotación: [-180°, +180°] = 360° círculo
}

"""
JUSTIFICACIÓN TÉCNICA: ROTACIÓN CONTINUA vs DISCRETA
───────────────────────────────────────────────────

PROBLEMA CON OVERSAMPLING PURO:
- Tenemos N copias exactas de la misma imagen
- Sin aumentación: modelo memoriza la imagen exacta
- Resultado: bajo RMSE pero mal generaliza a imágenes nuevas

SOLUCIÓN: Data Augmentation Combinada
- Cada copia oversampleada se transforma aleatoriamente
- Modelo ve "variantes" de la misma imagen → aprende features reales

ROTACIÓN DISCRETA [0°, 90°, 180°, 270°]:
  Generador: Random choice entre 4 ángulos
  Variaciones únicas: 4 (finitas)
  Con 100 copias del mismo panel:
    → Máximo 4 imágenes distintas
    → Memorización plausible
  ❌ No recomendado para oversampling

ROTACIÓN CONTINUA [-180°, +180°]:
  Generador: Uniforme aleatorio en rango de 360°
  Variaciones únicas: ∞ (teóricamente infinitas)
  Con 100 copias del mismo panel:
    → 100 imágenes con ángulos diferentes
    → ~0% probabilidad de dos copias idénticas
    → Prevención garantizada de memorización
  ✅ Recomendado para oversampling

IMPLEMENTACIÓN:
  transforms.RandomRotation(degrees=180)
  ↓
  Rota con ángulo θ ~ Uniform(-180°, +180°)
  ↓
  Cubre círculo completo [0°, 360°)

FÍSICA: Un panel con suciedad rotado 47.3° vs 47.4° mantiene
el mismo patrón de pérdida de potencia → augmentación válida.
"""

# ============================================================
# ARCHIVOS Y LOGGING
# ============================================================

# Formato de mensajes de logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Nivel de intensidad del logging
LOG_LEVEL = 'INFO'  # Niveles: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Nombre del archivo de historial de training
# Contiene: epoch, train_rmse, val_rmse, val_mae, val_r2, learning_rate
TRAINING_LOG_NAME = 'training_log_v3.csv'

# Nombre del archivo de checkpoint (para reanudar entrenamiento)
# Contiene: model_state_dict, optimizer_state_dict, best_val_rmse, epoch
CHECKPOINT_NAME = 'checkpoint_v3.pth'

# Nombre del archivo del mejor modelo encontrado
# Se guarda cuando: val_rmse < best_val_rmse
BEST_MODEL_NAME = 'best_model_v3.pth'

# ============================================================
# CONFIGURACIÓN DE IMAGEN
# ============================================================

# Tamaño entrada de la red (archivos originales se resamplearán)
# Estándar en visión: 224×224 (ImageNet)
# Trade-off: Mayor tamaño = más precisión pero más GPU memory
IMG_SIZE = 224

# Número de canales (3 para RGB)
IMG_CHANNELS = 3

# ImageNet normalization parameters (pre-trained weights estándar)
# Estos valores son fijos para compatibilidad con modelos pre-entrenados
# Cálculo: media y desviación estándar del conjunto ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Canales: [R, G, B]
IMAGENET_STD = [0.229, 0.224, 0.225]   # Canales: [R, G, B]

# ============================================================
# DIAGNÓSTICO DEL MODELO
# ============================================================

# Habilitar reporting de predicciones fuera [0, 100]
# Util para detectar problemas en output (sin sigmoid = diagnóstico)
OUT_OF_BOUNDS_DIAGNOSTIC = True

# Límites del rango físicamente válido
OUT_OF_BOUNDS_MIN = 0     # Power loss mínimo: 0% (panel perfecto)
OUT_OF_BOUNDS_MAX = 100   # Power loss máximo: 100% (panel sin función)


