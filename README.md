# Instacart Recommender ML Test

Pipeline de Machine Learning para recomendación de productos usando el dataset de Instacart.
El proyecto incluye un modelo de Learning to Rank, embeddings de productos y una demo sencilla en Streamlit.

## Demo

Streamlit app:

```text
URL: https://instacart-recommender-ml-test-gdmih5estreamlit.app/
```

Repositorio:

```text
URL: https://github.com/samathacqs2/instacart-recommender-m
```

## Resumen del proyecto

Este proyecto fue desarrollado usando el dataset **Instacart Online Grocery Basket Analysis**. La idea principal fue construir un flujo simple de recomendación dividido en tres partes:

1. Un modelo de ranking para recomendar productos a un usuario.
2. Embeddings de productos para encontrar productos similares.
3. Una app en Streamlit para probar las recomendaciones en una demo.

El entrenamiento completo se realizó en Kaggle Notebooks. En este repositorio se incluye el código, los notebooks y una muestra reducida de artefactos para que la demo pueda ejecutarse sin cargar todo el dataset original.

## Dataset

Archivos principales utilizados del dataset de Instacart:

* `orders.csv`
* `order_products__prior.csv`
* `order_products__train.csv`
* `products.csv`
* `aisles.csv`
* `departments.csv`

Los archivos originales no se incluyen en el repositorio por su tamaño. Para la demo solo se incluyen artefactos procesados y reducidos.

## Estructura del repositorio

```text
app/
  streamlit_app.py

src/
  ranking/
    service.py
  embeddings/
    service.py
  agent/

notebooks/
  01-prepare-data.ipynb
  02-train-ranker.ipynb
  03-train-embeddings.ipynb
  04-export-artifacts.ipynb

artifacts_sample/
  data/
  ranking/
  embeddings/
  reports/

requirements.txt
README.md
```

## Notebooks

| Notebook                    | Descripción                                                               |
| --------------------------- | ------------------------------------------------------------------------- |
| `01-prepare-data.ipynb`     | Carga y prepara los datos de Instacart.                                   |
| `02-train-ranker.ipynb`     | Construye el dataset de ranking, entrena el modelo XGBRanker y lo evalúa. |
| `03-train-embeddings.ipynb` | Crea embeddings de productos y prueba búsqueda de productos similares.    |
| `04-export-artifacts.ipynb` | Exporta una muestra reducida de artefactos para la demo en Streamlit.     |

## Learning to Rank

Para la etapa de ranking se crearon pares usuario-producto con las siguientes etiquetas:

| Label | Significado                       |
| ----: | --------------------------------- |
|   `2` | Producto comprado y reordenado    |
|   `1` | Producto comprado por primera vez |
|   `0` | Producto negativo muestreado      |

Los productos negativos se tomaron del mismo departamento que los productos positivos, excluyendo productos que el usuario ya había comprado.

Variables usadas por el modelo:

* `user_product_frequency`
* `user_product_reordered`
* `product_global_popularity`
* `department_id`

El modelo utilizado fue `XGBRanker` con objetivo de ranking. La evaluación se realizó con `NDCG@10` y `MAP@10`, comparando también contra un baseline de popularidad.

Las métricas se guardan en:

```text
artifacts_sample/reports/ranking_metrics.json
```

## Embeddings de productos

Para la parte de embeddings se generaron vectores de productos con el objetivo de soportar búsqueda de productos similares.

El notebook incluye:

* embeddings por co-ocurrencia de productos,
* embeddings de texto usando nombre, aisle y department,
* búsqueda de productos similares,
* búsqueda por keyword,
* medición básica de latencia.

La demo usa un índice FAISS reducido almacenado en:

```text
artifacts_sample/embeddings/
```

## App de Streamlit

La app tiene cuatro secciones:

* `Agente demo`: recomienda productos para un usuario de la muestra.
* `Historial de usuario`: muestra el historial de compras del usuario.
* `Ejemplos embeddings`: muestra ejemplos de similitud y búsqueda por texto.
* `Métricas`: muestra métricas del modelo y latencia de embeddings.

Como es una demo, no todos los usuarios del dataset están disponibles. Los usuarios disponibles se cargan desde:

```text
artifacts_sample/data/user_history_sample.parquet
```

## Cómo ejecutar

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la app:

```bash
streamlit run app/streamlit_app.py
```

## Notas

Este proyecto es un prototipo para una prueba técnica. Por eso, la demo usa artefactos reducidos en lugar del dataset completo. El procesamiento y entrenamiento principal se realizó en Kaggle, y los artefactos exportados se incluyen solo para facilitar la ejecución y despliegue de la app.
