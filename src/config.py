"""
config.py - Configuración centralizada para DeepSolarEye v3.3

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
# HYPERPARÁMETROS DE ENTRENAMIENTO v3.3
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
#
# Historial de cambios:
#   v3.0: LR=0.001  → volatilidad excesiva (picos épocas 4, 9)
#   v3.1: LR=0.0003 → mejoró, pero warmup no evitó pico (ép.1: 20.04)
#   v3.2: LR=0.0003 + warmup 3 ép. → pico desplazado a ép.3 (17.15)
#   v3.3: LR=0.0001 → conservador, sin warmup necesario.
#         Con inyección directa (sin BN1d), no hay inestabilidad.
# ReduceLROnPlateau reducirá aún más si val_rmse no mejora
LEARNING_RATE = 0.0001

# Máximo de épocas permitidas
# CAMBIO v3.3: Aumentado de 100 a 150.
# Con LR=0.0001 (más bajo), el modelo converge más lento.
# ReduceLROnPlateau (factor=0.5) puede reducir ~5 veces:
#   0.0001 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6
# Cada reducción necesita ~7 épocas de paciencia + exploración.
# 150 épocas da margen holgado para convergencia + ES natural.
# GTX 1650: ~3 min/época → 150 épocas ≈ 7.5h (asumible).
MAX_EPOCHS = 150

# Warmup lineal: DESACTIVADO en v3.3
# HISTORIAL:
#   v3.2: Warmup 3 épocas para estabilizar BN1d de rama MLP.
#   v3.3: Rama MLP eliminada → BN1d ya no existe → warmup innecesario.
#         Además, LR=0.0001 es suficientemente bajo para un arranque
#         estable sin calentamiento progresivo.
WARMUP_EPOCHS = 0

# ----- Early Stopping y Scheduler -----------

# Early Stopping: Paciencia (épocas sin mejora antes de detener)
# CAMBIO v3.3: Aumentado de 12 a 15.
# Justificación: Con LR=0.0001 (más bajo), el modelo mejora más
# lentamente. 15 épocas da margen suficiente para que el scheduler
# reduzca LR y el modelo tenga oportunidad de recuperarse.
# Historial: v2=7, v3.0-v3.2=12, v3.3=15
ES_PATIENCE = 15

# ReduceLROnPlateau: Paciencia antes de reducir learning rate
# Debe ser MENOR que ES_PATIENCE para que scheduler actúe primero
# CAMBIO v3.3: Aumentado de 5 a 7 (coherente con LR bajo).
# Con LR=0.0001, cada plateau necesita más épocas para saturar.
# Patrón: Scheduler reduce LR → modelo se recupera → ES lo ve
SCHEDULER_PATIENCE = 7

# CAMBIO v3.1: Factor reducido de 0.1 a 0.5
# Justificación: Con LR inicial más bajo (0.0003), reducir 90% de golpe
# lleva demasiado rápido a LR casi nulo. Factor=0.5 permite:
#   1. Más oportunidades de escape de mínimos locales
#   2. Reducción gradual coherente con LR conservador
# Referencia: Smith (2017) "Cyclical Learning Rates"
SCHEDULER_FACTOR = 0.5  # Factor de reducción: LR_nuevo = LR_actual * factor

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
# Contiene: epoch, train_rmse, val_rmse, val_mae, val_r2, learning_rate,
#           rmse por categoría (v3.1+)
TRAINING_LOG_NAME = 'training_log_v3.3.csv'

# Nombre del archivo de checkpoint (para reanudar entrenamiento)
# Contiene: model_state_dict, optimizer_state_dict, best_val_rmse, epoch
CHECKPOINT_NAME = 'checkpoint_v3.3.pth'

# Nombre del archivo del mejor modelo encontrado
# Se guarda cuando: val_rmse < best_val_rmse
BEST_MODEL_NAME = 'best_model_v3.3.pth'

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

# ============================================================
# FEATURES AMBIENTALES (v3.3 - Inyección Directa)
# ============================================================

# ANÁLISIS DE VIABILIDAD (datos reales del dataset):
# ─────────────────────────────────────────────────
# Todos los datos son de Junio 2017 (un solo mes).
#
# Feature          Corr(power_loss)  Decisión    Justificación
# ──────────────── ──────────────── ────────── ──────────────────────────
# Irradiance       0.1027           ✅ INCLUIR  Medición física real, [0,1]
# Hour/Min/Sec     0.0454           ❌ EXCLUIR  Orden temporal, no solar
# Day              0.1962           ❌ EXCLUIR  Correlación espuria (ciclos limpieza)
# Month/Year       N/A              ❌ EXCLUIR  Constantes (Jun 2017)
#
# La irradiance ya está normalizada en [0.003, 1.006] en el dataset.
#
# EVOLUCIÓN DE LA RAMA AMBIENTAL:
# ──────────────────────────────
# v3.2: Rama MLP dedicada (1→32→64) + fusión (160→96)
#       Resultado: Overfitting (train_rmse < val_rmse), 17,664 params extra
#       Diagnóstico: MLP sobredimensionado para 1 feature con corr=0.10
#                    BN1d sobre 1 feature causó gap artificial train/val
#
# v3.3: Inyección directa (concat irradiance cruda al vector CNN)
#       Solo 1 parámetro extra (peso de irradiance en fc_final)
#       Sin BN1d, sin capas intermedias, sin fuente de overfitting
#       Justificación: Con corr=0.10, basta un bias condicional lineal

# Número de features ambientales inyectadas directamente
NUM_ENV_FEATURES = 1  # Solo irradiance

# Gradient Clipping: limita norma del gradiente para estabilidad
# Evidencia v3.1: spike época 28 (RMSE +2.51 en una época)
# Referencia: Pascanu, Mikolov & Bengio (2013)
GRAD_CLIP_MAX_NORM = 1.0


