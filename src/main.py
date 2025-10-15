from data_processing import load_data, clean_sales_data, clean_promos_data # Asegúrate de que este archivo exista y contenga las funciones
from feature_engineering import feature_engineering
from modeling import train_and_evaluate_model, calculate_inventory_levels, save_model
from perform_eda import perform_eda
import config
import pandas as pd

def main():
    """Función principal para ejecutar el pipeline de análisis."""
    # --- 1. Carga de Datos ---
    print(f"Cargando datos desde: {config.DATA_DIR}")
    
    try:
        df_sales, df_promos = load_data(config.SALES_PATH, config.PROMOS_PATH)
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el archivo de datos. Asegúrate de que la ruta es correcta.")
        print(f"Detalle: {e}")
        print("La estructura esperada es: tu_proyecto/data/archivo.csv y tu_proyecto/src/main.py")
        return

    # --- 2. Limpieza de Datos ---
    df_sales_clean = clean_sales_data(df_sales)
    df_promos_clean = clean_promos_data(df_promos)

    # --- 3. Análisis Exploratorio (Opcional) ---
    print("Realizando Análisis Exploratorio de Datos (EDA)...")
    perform_eda(df_sales_clean)

    # --- 4. Ingeniería de Características ---
    print("\nRealizando Ingeniería de Características...")
    df_featured = feature_engineering(df_sales_clean, df_promos_clean)
    print("Características creadas:", [col for col in df_featured.columns if 'lag' in col or 'descuento' in col])

    # --- 5. Modelado y Evaluación ---
    print("\nEntrenando y evaluando el modelo...")
    model, X_test, y_test, y_pred = train_and_evaluate_model(df_featured)
    save_model(model, config.MODEL_PATH)

    # --- 6. Conclusiones Logísticas ---
    calculate_inventory_levels(y_test, y_pred)

    # --- 7. Guardar Predicciones Detalladas ---
    print("\nGuardando predicciones detalladas...")
    # Reconstruimos un DataFrame con la información original y las predicciones
    df_predictions = df_featured.loc[X_test.index].copy()
    df_predictions['cantidad_predicha'] = y_pred
    
    # Seleccionamos columnas relevantes para el reporte
    report_cols = ['id_producto', 'Fecha', 'cantidad_vendida', 'cantidad_predicha', 'precio_unitario']
    df_report = df_predictions[report_cols]
    
    predictions_path = pd.io.common.build_path(config.DATA_DIR, 'predictions.csv')
    df_report.to_csv(predictions_path, index=False)
    print(f"Reporte de predicciones guardado en: {predictions_path}")


if __name__ == "__main__":
    # Para ejecutar el script, asegúrate de tener las dependencias instaladas:
    # pip install pandas numpy matplotlib seaborn scikit-learn
    main()