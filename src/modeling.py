import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import config

def train_and_evaluate_model(df):
    """Entrena un modelo de Gradient Boosting y evalúa su rendimiento."""
    # Usar variables desde el archivo de configuración
    X = df[config.FEATURES]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, **config.SPLIT_PARAMS)

    # Usar parámetros desde el archivo de configuración
    gb_model = GradientBoostingRegressor(**config.GB_PARAMS)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    print("--- Evaluación del Modelo Gradient Boosting ---")
    evaluate_performance(y_test, y_pred_gb)

    # Visualización
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Real', alpha=0.7)
    plt.plot(y_pred_gb, label='Predicción GB', alpha=0.7)
    plt.title('Demanda real vs. predicha (Gradient Boosting)')
    plt.legend()
    plt.show()

    return gb_model, X_test, y_test, y_pred_gb

def save_model(model, path):
    """Guarda el modelo entrenado en un archivo."""
    print(f"\nGuardando modelo en: {path}")
    joblib.dump(model, path)

def evaluate_performance(y_true, y_pred):
    """Calcula e imprime las métricas de rendimiento del modelo."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"  - MSE: {mse:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - R²: {r2:.2f}")

def calculate_inventory_levels(y_test, y_pred):
    """Calcula el stock de seguridad y el inventario óptimo."""
    error_std = np.std(y_test - y_pred)
    safety_stock = error_std * (config.LEAD_TIME_DAYS ** 0.5)
    optimal_inventory = y_pred.mean() + safety_stock

    print("\n--- Conclusión Logística ---")
    print(f"Stock de seguridad recomendado: {safety_stock:.2f} unidades")
    print(f"Inventario óptimo estimado: {optimal_inventory:.2f} unidades")
    print("Recomendación: ajustar niveles de inventario semanalmente según predicción y eventos externos.")
