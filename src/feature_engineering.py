import pandas as pd
import numpy as np

def feature_engineering(df_sales, df_promos):
    """Crea nuevas características combinando datos de ventas y promociones."""
    df = df_sales.copy()
    df = df.sort_values('Fecha')

    # 1. Características temporales
    df['mes'] = df['Fecha'].dt.month
    df['año'] = df['Fecha'].dt.year
    df['dia_semana'] = df['Fecha'].dt.dayofweek

    # 2. Características de promociones
    df['descuento_aplicado'] = 0.0
    df_promos['es_2x1'] = df_promos['tipo_promocion'].str.contains('2x1', case=False, na=False)

    for _, promo in df_promos.iterrows():
        mask = (df['Fecha'] >= promo['fecha_inicio']) & (df['Fecha'] <= promo['fecha_final'])
        if promo['es_2x1']:
            # Para 2x1, el descuento es del 50%
            df.loc[mask, 'descuento_aplicado'] = 0.5
        else:
            df.loc[mask, 'descuento_aplicado'] = promo['descuento_porcentaje']

    df['precio_con_descuento'] = df['precio_unitario'] * (1 - df['descuento_aplicado'])
    df['ingreso_descuento'] = df['cantidad_vendida'] * df['precio_con_descuento']

    # 3. Características de lag (ventas pasadas)
    df['mes_periodo'] = df['Fecha'].dt.to_period('M')
    df_lag = df.groupby(['id_producto', 'mes_periodo'])['cantidad_vendida'].sum().reset_index()

    for i in range(1, 4):
        df_lag[f'cantidad_vendida_lag{i}'] = df_lag.groupby('id_producto')['cantidad_vendida'].shift(i)

    df = df.merge(df_lag.drop('cantidad_vendida', axis=1), on=['id_producto', 'mes_periodo'], how='left')
    
    lag_cols = ['cantidad_vendida_lag1', 'cantidad_vendida_lag2', 'cantidad_vendida_lag3']
    df[lag_cols] = df[lag_cols].fillna(0)
    df = df.drop(columns=['mes_periodo'])

    return df