# c:\Users\hecto\OneDrive\Desktop\Proyecto\Proyecto_Optimizacion_Inventario\src\data_processing.py

import pandas as pd
import numpy as np

def load_data(sales_path, promos_path):
    """Carga los datos de ventas y promociones desde archivos CSV."""
    df_sales = pd.read_csv(sales_path)
    df_promos = pd.read_csv(promos_path)
    return df_sales, df_promos

def clean_sales_data(df):
    """Limpia y preprocesa el DataFrame de ventas de forma robusta."""
    # 1. Limpiar nombres de columnas
    df.columns = df.columns.str.strip()

    # 2. Manejo de fechas (LA SOLUCIÓN ESTÁ AQUÍ)
    # Imputar nulos en 'Fecha' antes de convertir
    df['Fecha'] = df['Fecha'].ffill()
    # Usar format='mixed' para manejar formatos inconsistentes (DD/MM/YYYY y MM/DD/YYYY)
    # dayfirst=True resuelve ambigüedades como 01/02/2023 a favor del día primero.
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='mixed', dayfirst=True)

    # 3. Imputación de nulos en otras columnas
    df['id_producto'] = df['id_producto'].fillna(df['id_producto'].mode()[0])
    df['categoria_producto'] = df['categoria_producto'].fillna(df['categoria_producto'].mode()[0])
    
    # Imputar nulos en columnas numéricas con la media o mediana
    for col in ['cantidad_vendida', 'stock_inicial', 'stock_final', 'costo_adquisicion']:
        df[col] = df[col].fillna(df[col].mean())
    df['precio_unitario'] = df['precio_unitario'].fillna(df['precio_unitario'].median())

    # 4. Conversión de tipos de datos
    numeric_cols = ['cantidad_vendida', 'precio_unitario', 'stock_inicial', 'stock_final', 'costo_adquisicion']
    df[numeric_cols] = df[numeric_cols].astype(int)

    # 5. Eliminar columnas innecesarias si existen
    if 'ubicacion_almacen' in df.columns:
        df = df.drop('ubicacion_almacen', axis=1)
        
    return df

def clean_promos_data(df):
    """Limpia y preprocesa el DataFrame de promociones."""
    df.columns = df.columns.str.strip()
    df.rename(columns={'fecha': 'fecha_inicio'}, inplace=True)

    # Imputación de nulos
    df['evento_especial'].fillna('No', inplace=True)
    df['duracion_dias'].fillna(0, inplace=True)
    df['descuento_porcentaje'].fillna(0, inplace=True)
    df['tipo_promocion'].fillna('Sin promocion', inplace=True)

    # Manejo de fechas
    df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'], format='mixed', dayfirst=True, errors='coerce')
    df['fecha_inicio'].fillna(method='ffill', inplace=True) # Rellenar fechas nulas
    
    # Corrección de datos
    df['descuento_porcentaje'] /= 100
    df.loc[df['evento_especial'] == 'No', 'descuento_porcentaje'] = 0
    df['duracion_dias'] = df['duracion_dias'].astype(int)
    
    # Crear fecha final
    df['fecha_final'] = df['fecha_inicio'] + pd.to_timedelta(df['duracion_dias'], unit='D')
    
    return df
