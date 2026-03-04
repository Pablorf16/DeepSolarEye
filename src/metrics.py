"""
metrics.py - Métricas completas para evaluación de DeepSolarEye

Responde a los requerimientos del tutor:
1. Cálculo de RMSE (no solo MSE)
2. Soporte para Mean Percentage Error (MPE)
3. Interpretación física de cada métrica
4. Unidades claras
"""

import torch
import numpy as np
from typing import Tuple

class RegressionMetrics:
    """
    Calcula métricas de regresión estándar con interpretación física.
    
    UNIDADES:
    - Power Loss: porcentaje (0-100%)
    - Todas las métricas en las mismas unidades: puntos de porcentaje
    
    INTERPRETACIÓN:
    - MAE = Error Absoluto Promedio: en promedio, ¿cuántos puntos de % erras?
    - RMSE = Raíz del Error Cuadrático: penaliza errores grandes
    - MAPE = Error Porcentual Absoluto Medio: error RELATIVO (%, no absoluto)
    """
    
    @staticmethod
    def calculate_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Mean Squared Error
        
        Input: y_true, y_pred ∈ [0, 100] (puntos de porcentaje)
        Output: MSE (unidades: porcentaje²)
        
        Interpretación: Penaliza mucho los errores grandes
        """
        mse = torch.mean((y_pred - y_true) ** 2).item()
        return mse
    
    @staticmethod
    def calculate_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Root Mean Squared Error
        
        Input: y_true, y_pred ∈ [0, 100]
        Output: RMSE (unidades: puntos de porcentaje)
        
        Interpretación: "En promedio, los errores tienen magnitud RMSE%"
        Ejemplo: RMSE=8.4% significa que típicamente predices ±8.4 puntos de porcentaje
        """
        mse = RegressionMetrics.calculate_mse(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse
    
    @staticmethod
    def calculate_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Mean Absolute Error
        
        Input: y_true, y_pred ∈ [0, 100]
        Output: MAE (unidades: puntos de porcentaje)
        
        Interpretación: Error promedio sin penalizar magnitud
        Ejemplo: MAE=4.87% significa que en promedio erras 4.87 puntos de porcentaje
        """
        mae = torch.mean(torch.abs(y_pred - y_true)).item()
        return mae
    
    @staticmethod
    def calculate_mape(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> float:
        """
        Mean Absolute Percentage Error
        
        Input: y_true, y_pred ∈ [0, 100]
        Output: MAPE (unidades: %)
        
        Fórmula: MAPE = mean(|y_pred - y_true| / |y_true| × 100)
        
        Interpretación: Error RELATIVO.
        - Errar 5% en un panel limpio (real=5%) = 100% error relativo
        - Errar 5% en un panel sucio (real=80%) = 6.25% error relativo
        
        Problema: Undefined cuando y_true = 0. Usamos epsilon para evitarlo.
        """
        # Evitar división por cero
        denominator = torch.clamp(torch.abs(y_true), min=epsilon)
        percentage_errors = torch.abs((y_pred - y_true) / denominator) * 100
        mape = torch.mean(percentage_errors).item()
        return mape
    
    @staticmethod
    def calculate_r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Coeficiente de Determinación (R²)
        
        Input: y_true, y_pred
        Output: R² ∈ [-∞, 1]
        
        Interpretación:
        - R² = 1: Predicción perfecta
        - R² = 0: Predicción tan buena como predecir la media
        - R² < 0: Peor que predecir la media (¡muy mal!)
        
        Fórmula: R² = 1 - (SS_res / SS_tot)
        """
        y_mean = torch.mean(y_true)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_mean) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r_squared = 1 - (ss_res / ss_tot).item()
        return r_squared
    
    @staticmethod
    def get_all_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Calcula todas las métricas de una sola vez"""
        return {
            'mse': RegressionMetrics.calculate_mse(y_true, y_pred),
            'rmse': RegressionMetrics.calculate_rmse(y_true, y_pred),
            'mae': RegressionMetrics.calculate_mae(y_true, y_pred),
            'mape': RegressionMetrics.calculate_mape(y_true, y_pred),
            'r_squared': RegressionMetrics.calculate_r_squared(y_true, y_pred),
        }
    
    @staticmethod
    def format_metrics(metrics: dict) -> str:
        """Formatea las métricas para impresión legible"""
        return (
            f"MSE:  {metrics['mse']:.4f} (puntos²)\n"
            f"RMSE: {metrics['rmse']:.4f} (puntos de %)\n"
            f"MAE:  {metrics['mae']:.4f} (puntos de %)\n"
            f"MAPE: {metrics['mape']:.4f} (%)\n"
            f"R²:   {metrics['r_squared']:.4f}"
        )


class PredictionBounds:
    """
    Utilidades para manejar predicciones fuera del rango [0, 100].
    
    Como removimos sigmoid, el modelo puede predecir cualquier valor.
    Esto es BUENO para diagnosis, pero hay que decidir cómo reportar.
    """
    
    @staticmethod
    def clip_predictions(predictions: torch.Tensor, min_val: float = 0, max_val: float = 100) -> torch.Tensor:
        """
        Clip de predicciones al rango [min_val, max_val].
        
        NOTA: Esto se hace DESPUÉS de calcular el loss, no antes.
        El loss ve los valores reales para penalizar predicciones fuera de rango.
        """
        return torch.clamp(predictions, min=min_val, max=max_val)
    
    @staticmethod
    def get_out_of_bounds_count(predictions: torch.Tensor, min_val: float = 0, max_val: float = 100) -> Tuple[int, int]:
        """
        Cuenta cuántas predicciones están fuera del rango válido.
        
        Return: (count_below_min, count_above_max)
        
        DIAGNÓSTICO: Si muchas están fuera de rango, el modelo está aprendiendo mal.
        """
        below = (predictions < min_val).sum().item()
        above = (predictions > max_val).sum().item()
        return below, above
    
    @staticmethod
    def print_bound_diagnostics(predictions: torch.Tensor, min_val: float = 0, max_val: float = 100):
        """Diagnóstico de predicciones fuera de rango"""
        below, above = PredictionBounds.get_out_of_bounds_count(predictions, min_val, max_val)
        total = len(predictions)
        
        print(f"\n🔍 Diagnóstico de Rango [0, 100]:")
        print(f"   Predicciones < {min_val}:  {below:4d} ({100*below/total:5.2f}%)")
        print(f"   Predicciones > {max_val}: {above:4d} ({100*above/total:5.2f}%)")
        print(f"   En rango:         {total-below-above:4d} ({100*(total-below-above)/total:5.2f}%)")
        
        if below > 0 or above > 0:
            print(f"   ⚠️ ADVERTENCIA: {(100*(below+above)/total):.1f}% de predicciones fuera de rango.")
            print(f"      Esto es NORMAL antes de entrenamiento, pero debe disminuir.")
            print(f"      Si persiste, el modelo tiene problemas de aprendizaje.")
