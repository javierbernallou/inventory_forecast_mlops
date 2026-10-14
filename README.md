# Inventory Forecast MLOps

## Descripción general
Proyecto enfocado en el análisis del dataset M5 y en la preparación de un flujo de machine learning para la predicción de demanda e inventario. El repositorio combina análisis exploratorio, preparación de features y experimentación con modelos predictivos orientados a recomendaciones de stock.

## Objetivo
Apoyar la toma de decisiones de inventario mediante la estimación de la demanda futura y la generación de recomendaciones operativas basadas en datos históricos y métricas de forecasting.

## Stack principal
- Lenguaje: Python
- Entorno de análisis: Jupyter Notebooks
- Bibliotecas principales: pandas, NumPy, scikit-learn, y librerías de forecasting según el modelo implementado
- Enfoque MLOps: versionado, trazabilidad y automatización de experimentos

## Estructura del proyecto
```bash
notebooks/           EDA y experimentos de modelado
data/                Datasets o instrucciones para su obtención
authors/             Archivos de soporte y código auxiliar
src/                 Utilidades de preprocesado y lógica de negocio
requirements.txt     Dependencias del proyecto
README.md            Documentación del repositorio
```

## Cómo ejecutar
1. Clonar el repositorio.
2. Crear entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Abrir los notebooks y ejecutar el flujo en orden:
   - análisis exploratorio,
   - preparación de features,
   - entrenamiento del modelo,
   - evaluación y recomendaciones.

## Caso de uso
Ideal para proyectos de forecasting de demanda y optimización de inventarios en entornos retail o de distribución, donde es necesario anticipar la demanda con precisión y ajustar compras y stock de forma inteligente.

## Recomendaciones de buen uso
- versionar modelos y datasets,
- guardar métricas de error para comparar experimentos,
- automatizar el pipeline para reproducibilidad.
