# Instacart Recommender ML Test

Machine Learning pipeline for product recommendation using the Instacart dataset.
The project includes a Learning to Rank model, product embeddings, and a small Streamlit demo.

## Demo

Streamlit app:

```text
TODO: add Streamlit link here
```

Repository:

```text
TODO: add GitHub repo link here
```

## Project summary

This project was built using the Instacart Online Grocery Basket Analysis dataset. The main idea was to create a simple recommendation flow with three parts:

1. A ranking model to recommend products for a user.
2. Product embeddings to find similar products.
3. A Streamlit app to test the recommendations in a small demo.

The full training was done in Kaggle Notebooks. The repository includes the code, notebooks, and a reduced set of artifacts so the demo can run without the full dataset.

## Dataset

Main files used from the Instacart dataset:

* `orders.csv`
* `order_products__prior.csv`
* `order_products__train.csv`
* `products.csv`
* `aisles.csv`
* `departments.csv`

The original CSV files are not included in this repository because they are large. Only processed sample artifacts are included for the demo.

## Repository structure

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

| Notebook                    | Purpose                                                                   |
| --------------------------- | ------------------------------------------------------------------------- |
| `01-prepare-data.ipynb`     | Loads and prepares the Instacart data.                                    |
| `02-train-ranker.ipynb`     | Builds the ranking dataset, trains the XGBRanker model, and evaluates it. |
| `03-train-embeddings.ipynb` | Creates product embeddings and tests product similarity search.           |
| `04-export-artifacts.ipynb` | Exports a smaller artifact package for the Streamlit demo.                |

## Learning to Rank

For the ranking stage, I created user-product pairs and assigned labels:

| Label | Meaning                               |
| ----: | ------------------------------------- |
|   `2` | Product was bought and reordered      |
|   `1` | Product was bought for the first time |
|   `0` | Negative sampled product              |

The negative products were sampled from the same department as the positive product, excluding products the user had already bought.

The model uses these features:

* `user_product_frequency`
* `user_product_reordered`
* `product_global_popularity`
* `department_id`

The model used was `XGBRanker` with a ranking objective. The evaluation was done using `NDCG@10` and `MAP@10`. A popularity baseline was also included for comparison.

The metrics are saved in:

```text
artifacts_sample/reports/ranking_metrics.json
```

## Product embeddings

For the embedding part, I created product vectors to support similarity search.

The notebook includes:

* product co-occurrence embeddings,
* text embeddings using product name, aisle, and department,
* product similarity search,
* keyword-based search,
* latency check.

The demo uses a reduced FAISS index stored in:

```text
artifacts_sample/embeddings/
```

## Streamlit app

The Streamlit app has four sections:

* `Agente demo`: recommends products for a selected demo user.
* `Historial de usuario`: shows the user's purchase history.
* `Ejemplos embeddings`: shows product similarity and keyword search examples.
* `Métricas`: shows ranking metrics and embedding latency.

Since the app is only a demo, it uses a sample of users. The available users are loaded from:

```text
artifacts_sample/data/user_history_sample.parquet
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

## Notes

This is a technical test prototype, so the demo uses reduced artifacts instead of the full dataset. The full data preparation and model training were done in Kaggle, and the exported sample artifacts are included only to make the app easier to run and deploy.
