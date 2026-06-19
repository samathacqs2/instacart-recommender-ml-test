from pathlib import Path

import joblib
import pandas as pd


class RankingService:
    def __init__(self, artifacts_dir: str = "artifacts_sample"):
        self.artifacts_dir = Path(artifacts_dir)

        self.model = joblib.load(
            self.artifacts_dir / "ranking" / "xgb_ranker.joblib"
        )

        self.user_history = pd.read_parquet(
            self.artifacts_dir / "data" / "user_history_sample.parquet"
        )

        self.product_catalog = pd.read_parquet(
            self.artifacts_dir / "data" / "product_catalog.parquet"
        )

        self.product_popularity = pd.read_parquet(
            self.artifacts_dir / "data" / "product_popularity.parquet"
        )

        self.product_catalog_min = (
            self.product_catalog
            .drop_duplicates("product_id")
            .set_index("product_id")
        )

        self.product_popularity_map = (
            self.product_popularity
            .set_index("product_id")["product_global_popularity"]
            .to_dict()
        )

        self.features = [
            "user_product_frequency",
            "user_product_reordered",
            "product_global_popularity",
            "department_id",
        ]

    def get_user_history(self, user_id: int, limit: int = 10):
        user_id = int(user_id)

        history = self.user_history[
            self.user_history["user_id"] == user_id
        ].copy()

        if history.empty:
            return []

        result = (
            history
            .groupby(["product_id", "product_name", "aisle", "department"])
            .agg(
                purchase_count=("order_id", "nunique"),
                reordered_count=("reordered", "sum"),
            )
            .reset_index()
            .sort_values(
                ["purchase_count", "reordered_count"],
                ascending=False
            )
            .head(limit)
        )

        return result.to_dict(orient="records")

    def build_features(self, user_id: int, product_id: int):
        user_id = int(user_id)
        product_id = int(product_id)

        history = self.user_history[
            self.user_history["user_id"] == user_id
        ]

        product_history = history[
            history["product_id"] == product_id
        ]

        user_product_frequency = product_history["order_id"].nunique()

        if product_history.empty:
            user_product_reordered = 0
        else:
            user_product_reordered = int(product_history["reordered"].max())

        product_global_popularity = int(
            self.product_popularity_map.get(product_id, 0)
        )

        if product_id in self.product_catalog_min.index:
            department_id = int(
                self.product_catalog_min.loc[product_id]["department_id"]
            )
            product_name = self.product_catalog_min.loc[product_id]["product_name"]
        else:
            department_id = -1
            product_name = None

        return {
            "user_product_frequency": int(user_product_frequency),
            "user_product_reordered": int(user_product_reordered),
            "product_global_popularity": int(product_global_popularity),
            "department_id": int(department_id),
            "product_name": product_name,
        }

    def predict_reorder(self, user_id: int, product_id: int):
        feature_dict = self.build_features(user_id, product_id)

        X = pd.DataFrame([{
            "user_product_frequency": feature_dict["user_product_frequency"],
            "user_product_reordered": feature_dict["user_product_reordered"],
            "product_global_popularity": feature_dict["product_global_popularity"],
            "department_id": feature_dict["department_id"],
        }])

        score = float(self.model.predict(X[self.features])[0])

        return {
            "user_id": int(user_id),
            "product_id": int(product_id),
            "product_name": feature_dict["product_name"],
            "ranking_score": score,
            "features": {
                "user_product_frequency": feature_dict["user_product_frequency"],
                "user_product_reordered": feature_dict["user_product_reordered"],
                "product_global_popularity": feature_dict["product_global_popularity"],
                "department_id": feature_dict["department_id"],
            }
        }

    def user_exists(self, user_id: int):
        return int(user_id) in set(self.user_history["user_id"].unique())