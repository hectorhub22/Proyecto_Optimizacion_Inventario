# Optimización de Inventario y Predicción de Demanda

Este proyecto es un caso de estudio completo que demuestra cómo utilizar Machine Learning para predecir la demanda de productos en un entorno de retail, con el objetivo de optimizar los niveles de inventario, reducir costos y minimizar las ventas perdidas.

**Visita el caso de estudio completo en la página del portafolio.**

## El Problema de Negocio

Una empresa de retail enfrentaba pérdidas significativas debido a dos problemas comunes:
-   **Quiebres de stock** en productos de alta demanda, resultando en ventas perdidas.
-   **Exceso de inventario** en productos de baja rotación, aumentando los costos de almacenamiento.

Carecían de un método sistemático para prever la demanda futura y optimizar sus niveles de inventario.

## Solución Implementada

Se desarrolló un pipeline de análisis de datos de extremo a extremo en Python que:
1.  **Ingiere y Limpia** datos históricos de ventas y promociones.
2.  **Crea Características Relevantes** (features) para capturar tendencias y estacionalidad.
3.  **Entrena un Modelo** de Machine Learning (Gradient Boosting) para predecir la demanda futura.
4.  **Calcula Recomendaciones** de negocio, como el stock de seguridad y el inventario óptimo.

## Resultados Clave

-   **🎯 Precisión del Modelo:** El sistema predice la demanda con un **R² de 0.89**.
-   **📉 Reducción de Errores:** El error promedio de predicción (RMSE) es de solo **11 unidades**, permitiendo una planificación precisa.
-   **💰 Impacto Estimado:** Reducción de costos de almacenamiento de hasta un **15%** y disminución de ventas perdidas en un **20%**.

## Estructura del Proyecto

```
.
├── data/                 # Archivos CSV con los datos brutos
├── models/               # Modelo entrenado guardado (.joblib)
├── notebooks/            # Jupyter notebooks para exploración inicial
├── portfolio-web/        # Código fuente del sitio web del portafolio (Quarto)
├── src/                  # Código fuente modularizado del pipeline
│   ├── config.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── main.py
│   ├── modeling.py
│   └── perform_eda.py
└── requirements.txt      # Dependencias del proyecto
```

## Cómo Ejecutar el Proyecto

1.  Clona el repositorio:
    ```bash
    git clone https://github.com/hectorhub22/Proyecto_Optimizacion_Inventario.git
    cd Proyecto_Optimizacion_Inventario
    ```
2.  Crea un entorno virtual e instala las dependencias:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  Ejecuta el pipeline de entrenamiento:
    ```bash
    python src/main.py
    ```

## Herramientas Utilizadas

Python | Pandas | Scikit-learn | Matplotlib | Quarto
