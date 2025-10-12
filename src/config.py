"""
Archivo de Configuración

Este archivo centraliza todas las rutas, parámetros y variables
utilizadas en el proyecto de predicción de demanda.
"""
import os

# --- Rutas del Proyecto ---
# Directorio base del proyecto (un nivel arriba de 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Asegurarse de que el directorio de modelos exista
os.makedirs(MODELS_DIR, exist_ok=True)

# Rutas a los archivos de datos y modelo
SALES_PATH = os.path.join(DATA_DIR, 'Ventas Históricas y Nivel de Inventario.csv')
PROMOS_PATH = os.path.join(DATA_DIR, 'Promociones y Eventos Externos.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'demand_model.joblib')

# --- Variables para el Modelo ---
FEATURES = [
    'cantidad_vendida_lag1', 'cantidad_vendida_lag2', 'cantidad_vendida_lag3',
    'descuento_aplicado', 'dia_semana', 'mes', 'año',
    'stock_inicial', 'precio_unitario'
]
TARGET = 'cantidad_vendida'

# --- Parámetros del Modelo ---
GB_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
SPLIT_PARAMS = {'test_size': 0.2, 'random_state': 42}
LEAD_TIME_DAYS = 7