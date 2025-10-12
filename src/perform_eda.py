import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """Realiza y muestra gráficos de análisis exploratorio de datos."""
    df_eda = df.copy()
    df_eda['mes'] = df_eda['Fecha'].dt.month
    df_eda['dia_semana'] = df_eda['Fecha'].dt.dayofweek
    df_eda['año'] = df_eda['Fecha'].dt.year

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_eda, x='Fecha', y='cantidad_vendida')
    plt.title('Evolución de la demanda a lo largo del tiempo')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.barplot(x='mes', y='cantidad_vendida', data=df_eda, estimator='mean')
    plt.title('Demanda promedio por mes')
    plt.show()