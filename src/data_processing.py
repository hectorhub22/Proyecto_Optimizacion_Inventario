import pandas as pd
import numpy as np

def load_data(sales_path, promos_path):
    """Carga los datos de ventas y promociones desde archivos CSV."""
    df_sales = pd.read_csv(sales_path)
    df_promos = pd.read_csv(promos_path)
    return df_sales, df_promos

def clean_sales_data(df):
    """Limpia y preprocesa el DataFrame de ventas."""
    df.columns = df.columns.str.strip()

    # Imputación de nulos
    df['Fecha'] = df['Fecha'].ffill()
    df['id_producto'] = df['id_producto'].fillna(df['id_producto'].mode()[0])
    df['categoria_producto'] = df['categoria_producto'].fillna(df['categoria_producto'].mode()[0])
    
    for col in ['cantidad_vendida', 'stock_inicial', 'stock_final', 'costo_adquisicion']:
        df[col] = df[col].fillna(df[col].mean())
    df['precio_unitario'] = df['precio_unitario'].fillna(df['precio_unitario'].median())

    # Conversión de tipos
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
    numeric_cols = ['cantidad_vendida', 'precio_unitario', 'stock_inicial', 'stock_final', 'costo_adquisicion']
    df[numeric_cols] = df[numeric_cols].astype(int)

    # Limpieza de texto
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Eliminar y añadir columnas
    df = df.drop('ubicacion_almacen', axis=1, errors='ignore')
    df['ubicacion_almacen_latitud'] = '-33.45422901394828'
    df['ubicacion_almacen_longitud'] = '-70.78248510575368'
    
    return df

def clean_promos_data(df):
    """Limpia y preprocesa el DataFrame de promociones."""
    df.columns = df.columns.str.strip()
    
    # Imputación de nulos
    df['evento_especial'].fillna('No', inplace=True)
    df['duracion_dias'].fillna(0, inplace=True)
    df['descuento_porcentaje'].fillna(0, inplace=True)
    df['tipo_promocion'].fillna('Sin promocion', inplace=True)
    df['evento_deportivo'].fillna('No evento deportivo', inplace=True)
    df['temporada_alta'].fillna('No temporada alta', inplace=True)
    df['promocion_destacada'].fillna('False', inplace=True)

    # Corrección de datos
    df['descuento_porcentaje'] /= 100
    df.loc[df['evento_especial'] == 'No', 'descuento_porcentaje'] = 0

    # Manejo de fechas
    df.rename(columns={'fecha': 'fecha_inicio'}, inplace=True)
    df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'], format='%d/%m/%Y', errors='coerce')
    df['fecha_inicio'].fillna(df['fecha_inicio'].mean(), inplace=True)
    
    df['duracion_dias'] = df['duracion_dias'].astype(int)
    df.loc[df['duracion_dias'] >= 10, 'duracion_dias'] = np.random.randint(3, 10, size=(df['duracion_dias'] >= 10).sum())
    
    df['fecha_final'] = df['fecha_inicio'] + pd.to_timedelta(df['duracion_dias'], unit='D')

    # Filtrar promociones no informativas
    condition = (df['evento_especial'] == 'No') & \
                (df['descuento_porcentaje'] == 0) & \
                (df['tipo_promocion'].isin(['Sin promocion', 'Descuento']))
    df.drop(df[condition].index, inplace=True)
    
    df.drop_duplicates(subset=['fecha_inicio'], inplace=True)
    
    return df