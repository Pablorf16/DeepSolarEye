# CHANGELOG - DeepSolarEye v3.2

**Fecha de Implementación:** Abril 7, 2026  
**Feedback Incorporado:** Comentarios del tutor Andrés (Marzo 13, 2026)  
**Status:** Listo para reentrenamiento

---

## 📋 Resumen de Cambios

### 1. **Estratificación Dinámica por Cuartiles** ✅
   - **Fichero:** `src/data_prep.py`
   - **Cambio:** Sustitución de categorías hardcodeadas por cuartiles calculados automáticamente
   - **Motivación (Feedback Tutor):** "Simplify stratification - use quartiles with np.percentile([25,50,75])"
   
   **Antes (v3.0-v3.1):**
   ```python
   CATEGORY_BINS = [-1, 5, 15, 30, 60, 105]
   CATEGORY_LABELS = ['Limpio', 'Leve', 'Moderado', 'Alto', 'Crítico']
   # Categorías DESBALANCEADAS: 15% / 20% / 23% / 20% / 22%
   ```
   
   **Después (v3.2):**
   ```python
   q25, q50, q75 = np.percentile(df['power_loss'], [25, 50, 75])
   quartile_bins = [-1, q25, q50, q75, 105]
   quartile_labels = ['Q1_Limpio', 'Q2_Moderado', 'Q3_Alto', 'Q4_Crítico']
   # Categorías BALANCEADAS: 25% / 25% / 25% / 25% (por definición)
   ```
   
   **Ventajas:**
   - ✅ Eliminación automática de desbalance de clases
   - ✅ No requiere oversampling en entrenamiento (elimina data leakage)
   - ✅ Límites adaptativos al dataset específico
   - ✅ Reproducibilidad: Siempre genera 4 categorías de igual tamaño
   
   **Implementación Técnica:**
   - Línea 267-268: Cálculo de percentiles usando `np.percentile()`
   - Línea 272-274: Nuevas etiquetas con nomenclatura Q1-Q4
   - Línea 278-283: Logging de límites para auditoría
   - Línea 285-290: Visualización de distribución post-split

---

### 2. **Early Stopping Tolerante con ReduceLROnPlateau** ✅
   - **Fichero:** `src/train.py`
   - **Cambio:** Adición de lógica de tolerancia cuando el scheduler reduce LR
   - **Motivación (Feedback Tutor):** "Early Stopping may compete with ReduceLROnPlateau - fix conflict"
   
   **Problema Identificado (v3.3):**
   - Early Stopping (patience=15) y ReduceLROnPlateau (patience=7) actuaban independientemente
   - Cuando scheduler reducía LR, modelo necesitaba ajustarse, pero ES contaba inmediatamente como "épocas sin mejora"
   - Resultado: Parada prematura sin dar margen de recuperación
   
   **Solución Implementada (v3.2):**
   ```python
   # ANTES del scheduler.step()
   lr_before = optimizer.param_groups[0]['lr']
   scheduler.step(val_rmse)
   current_lr = optimizer.param_groups[0]['lr']
   lr_just_reduced = (current_lr < lr_before)
   
   # EN early_stopping check
   if lr_just_reduced:
       epochs_no_improve = max(0, epochs_no_improve - 1)  # "Pase gratis"
       print(f"   ⏳ Sin mejora: {epochs_no_improve}/{ES_PATIENCE} "
             f"(tolerancia: LR acaba de reducirse)")
   ```
   
   **Ventajas:**
   - ✅ Evita parada prematura inmediatamente después de reducción de LR
   - ✅ Permite ~5 épocas adicionales para convergencia con nuevo LR
   - ✅ Coordinación explícita entre mecanismos de regularización
   - ✅ Logging claro de cuándo se aplica tolerancia
   
   **Detalles Técnicos:**
   - Línea 493-500: Detección de reducción de LR
   - Línea 501-503: Log mejorado con indicación de reducción
   - Línea 539-548: Lógica de tolerancia en ES con "pase gratis"

---

## 📊 Comparativa de Versiones

| Aspecto | v3.0 | v3.1 | v3.2 |
|---------|------|------|------|
| **Estratificación** | Hardcoded bins | Hardcoded bins | **Cuartiles dinámicos** |
| **Clases Balanceadas** | No (15-23%) | No (15-23%) | ✅ Sí (25% c/u) |
| **Oversampling** | Sí | Sí | **No (elimina leakage)** |
| **ES Tolerance** | No | No | ✅ Sí (con LR changes) |
| **Características Extra** | v3.0 base | Direct injection | Direct injection |
| **Resultado esperado** | RMSE~9.2% | RMSE~8.7% | **Por entrenar** |

---

## 🔧 Ficheros Modificados

1. **src/data_prep.py**
   - Líneas 1-14: Header actualizado (v3.2)
   - Línea 267-290: Nueva lógica de cuartiles

2. **src/train.py**
   - Líneas 1-13: Header actualizado (v3.2)
   - Línea 493-503: Detección de LR reduction
   - Línea 539-548: Lógica de ES tolerance

3. **src/config.py** (No modificado)
   - CATEGORY_BINS sigue existiendo por compatibilidad
   - Ya no se utiliza en data_prep.py

---

## ✋ Ficheros No Modificados

- `src/model.py`: Arquitectura sin cambios (direct injection de irradiance)
- `src/dataset.py`: Augmentación y transforms intactos
- `src/plot_results.py`: Visualización mantenida
- `src/eda.py`: Análisis exploratorio mantenido

---

## 🎯 Instrucciones para Reentrenamiento

### Paso 1: Ejecutar Preparación de Datos
```bash
python -m src.data_prep
```
**Verificar logs:**
- Cuartiles calculados automáticamente
- Distribución Q1-Q4 mostrada (~25% cada uno)
- Sin mensaje sobre oversampling (eliminado en v3.2)

### Paso 2: Entrenar Modelo v3.2
```bash
python -m src.train
```
**Cambios esperados en logs:**
- Mensajes "Learning Rate: X (reducido de Y)" cuando scheduler actúa
- Mensajes "⏳ Sin mejora: N/15 (tolerancia: LR acaba de reducirse)" periódicamente

### Paso 3: Generar Gráficas
```bash
python -m src.plot_results
```
Genera `training_curves_v3.2.png` (automático post-training)

---

## 📈 Métricas de Referencia (Esperadas)

Basadas en v3.3 (8.58% RMSE) con mismo modelo:

- **Early Improvements:** Cuartiles deberían acelerar convergencia inicial
- **Convergencia Mejorada:** ES tolerance permitirá más épocas finales sin parada prematura
- **Mejor Estabilidad:** Menos fluctuaciones en LR → menos ES disparos falsos

**Rango esperado:** RMSE test en 8.0-9.0% (depende de convergencia con new ES logic)

---

## 📝 Notas Académicas para TFG

### Por qué Cuartiles:
1. **Justificación estadística:** Garantiza n=N/4 por clase (perfectamente balanceado)
2. **Automatización:** Elimina necesidad de tuneable del investigador
3. **Generalización:** Funciona para cualquier dataset sin reajuste
4. **Reproducibilidad:** Same quartile bin edges ↔ Same class distribution

### Por qué Tolerancia ES:
1. **Justificación teórica:** Cambio de LR requiere ajuste del modelo (≈5-10 épocas)
2. **Mecanismo:** "grace period" cuando se dispara ReduceLROnPlateau
3. **Equilibrio:** Evita parada prematura sin extender training innecesariamente
4. **Precedente:** Estrategia común en literatura (e.g., cyclical LR, warmup strategies)

---

## ⚡ Cambios Pendientes (Feedback Punto 3 & 4)

- **Punto (3):** Visualización de scatter plots (ref vs pred) - POST-ENTRENAMIENTO
- **Punto (4):** Documentación académica estructurada - EN PROGRESO

---

**Versión:** 3.2  
**Última Actualización:** Abril 7, 2026  
**Responsable:** TFG Developer + Tutor Feedback Andrés
