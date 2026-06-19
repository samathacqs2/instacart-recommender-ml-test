from pathlib import Path

import faiss
import numpy as np
import pandas as pd


class EmbeddingService:
    def __init__(self, artifacts_dir: str = "artifacts_sample"):
        self.artifacts_dir = Path(artifacts_dir)

        embeddings_dir = self.artifacts_dir / "embeddings"

        self.index = faiss.read_index(
            str(embeddings_dir / "text_products_demo.faiss")
        )

        self.product_ids = np.load(
            embeddings_dir / "text_product_ids_demo.npy"
        )

        self.catalog = pd.read_parquet(
            embeddings_dir / "product_catalog_embeddings_demo.parquet"
        )

        self.catalog_by_product_id = (
            self.catalog
            .drop_duplicates("product_id")
            .set_index("product_id")
        )

    def search_by_keyword_vector(self, query_vector: np.ndarray, k: int = 10):
        """
        Busca usando un vector ya generado.
        Para la demo sin LLM, esta función será usada internamente
        si luego agregas SentenceTransformer en Streamlit.
        """
        query_vector = query_vector.astype("float32")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self.index.search(query_vector, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            product_id = int(self.product_ids[int(idx)])

            if product_id not in self.catalog_by_product_id.index:
                continue

            row = self.catalog_by_product_id.loc[product_id]

            results.append({
                "product_id": product_id,
                "product_name": row["product_name"],
                "aisle": row["aisle"],
                "department": row["department"],
                "score": float(score),
            })

        return results

    def get_product_by_id(self, product_id: int):
        product_id = int(product_id)

        if product_id not in self.catalog_by_product_id.index:
            return None

        row = self.catalog_by_product_id.loc[product_id]

        return {
            "product_id": product_id,
            "product_name": row["product_name"],
            "aisle": row["aisle"],
            "department": row["department"],
        }