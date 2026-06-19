# Artifacts Sample

This folder contains lightweight artifacts for the Streamlit demo.

## Contents

### Ranking
- `ranking/xgb_ranker.joblib`: trained XGBRanker model.
- `reports/ranking_metrics.json`: NDCG@10 and MAP@10 results.

### Embeddings
- `embeddings/text_products_demo.faiss`: FAISS index for semantic product search.
- `embeddings/text_embeddings_demo.npy`: normalized text embeddings.
- `embeddings/text_product_ids_demo.npy`: product IDs aligned with the embeddings.
- `embeddings/product_catalog_embeddings_demo.parquet`: product metadata for the demo index.

### Data
- `data/user_history_sample.parquet`: sampled user purchase history for demo users.
- `data/product_catalog.parquet`: product catalog with names, aisles and departments.
- `data/product_popularity.parquet`: product global popularity.

## Notes

The full training was executed in Kaggle notebooks.  
This folder is reduced to keep the GitHub repository and Streamlit demo lightweight.

Demo users: 1000  
Demo products in semantic index: 15000
