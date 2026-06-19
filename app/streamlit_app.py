import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.ranking.service import RankingService


st.set_page_config(
    page_title="Instacart Recommender",
    layout="wide"
)


@st.cache_resource
def load_ranking_service():
    return RankingService(artifacts_dir=str(ROOT_DIR / "artifacts_sample"))


@st.cache_data
def load_reports():
    reports_dir = ROOT_DIR / "artifacts_sample" / "reports"

    with open(reports_dir / "ranking_metrics.json", "r") as f:
        ranking_metrics = json.load(f)

    with open(reports_dir / "embedding_latency.json", "r") as f:
        embedding_latency = json.load(f)

    with open(reports_dir / "embedding_examples.json", "r") as f:
        embedding_examples = json.load(f)

    return ranking_metrics, embedding_latency, embedding_examples


ranking_service = load_ranking_service()
ranking_metrics, embedding_latency, embedding_examples = load_reports()


st.title("Instacart Product Recommender")
st.caption(
    "Learning to Rank + Product Embeddings + Conversational Recommendation Demo"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Agente demo",
    "Historial de usuario",
    "Ejemplos embeddings",
    "Métricas"
])


with tab1:
    st.header("Agente demo de recomendación")

    st.write(
        "Esta demo usa los artefactos entrenados en Kaggle. "
        "El agente combina historial del usuario, productos populares y score del ranker."
    )

    user_id = st.number_input(
        "User ID",
        min_value=1,
        value=1,
        step=1
    )

    message = st.text_input(
        "Mensaje del usuario",
        value="Quiero armar una cesta saludable para el desayuno"
    )

    k = st.slider("Número de recomendaciones", 3, 15, 10)

    if st.button("Recomendar"):
        history = ranking_service.get_user_history(user_id, limit=10)

        if not history:
            st.warning(
                "Este usuario no está en la muestra de demo. "
                "Prueba con otro user_id o revisa la pestaña de historial."
            )
        else:
            # Candidatos simples para demo:
            # productos más populares + productos del historial
            popular = (
                ranking_service.product_popularity
                .head(100)
                ["product_id"]
                .astype(int)
                .tolist()
            )

            history_product_ids = [
                int(item["product_id"])
                for item in history
            ]

            candidate_ids = list(dict.fromkeys(history_product_ids + popular))

            scored = []

            for product_id in candidate_ids:
                pred = ranking_service.predict_reorder(user_id, product_id)

                if pred["product_name"] is None:
                    continue

                scored.append({
                    "product_id": pred["product_id"],
                    "product_name": pred["product_name"],
                    "ranking_score": pred["ranking_score"],
                    "user_product_frequency": pred["features"]["user_product_frequency"],
                    "user_product_reordered": pred["features"]["user_product_reordered"],
                    "product_global_popularity": pred["features"]["product_global_popularity"],
                })

            scored_df = (
                pd.DataFrame(scored)
                .sort_values("ranking_score", ascending=False)
                .head(k)
            )

            st.subheader("Respuesta del agente")

            st.write(
                f"Basado en el historial del usuario **{user_id}** y en el modelo de ranking, "
                f"te recomiendo estos productos para la intención: _{message}_"
            )

            st.dataframe(scored_df, use_container_width=True)

            st.info(
                "Nota: `ranking_score` es un score de relevancia del XGBRanker, "
                "no una probabilidad calibrada."
            )


with tab2:
    st.header("Historial de usuario")

    demo_user_id = st.number_input(
        "Consultar User ID",
        min_value=1,
        value=1,
        step=1,
        key="history_user_id"
    )

    if st.button("Ver historial"):
        history = ranking_service.get_user_history(demo_user_id, limit=20)

        if not history:
            st.warning("No hay historial para este usuario en la muestra de demo.")
        else:
            st.dataframe(pd.DataFrame(history), use_container_width=True)


with tab3:
    st.header("Ejemplos de embeddings")

    st.write(
        "Estos ejemplos fueron generados en Kaggle durante la Etapa 2."
    )

    st.subheader("Productos similares")
    similar_examples = embedding_examples.get("similar_products_examples", {})

    for name, example in similar_examples.items():
        st.markdown(f"### {name}")
        results = example.get("results", [])
        st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.subheader("Búsqueda por keyword")
    keyword_examples = embedding_examples.get("keyword_search_examples", {})

    for query, results in keyword_examples.items():
        st.markdown(f"### Query: `{query}`")
        st.dataframe(pd.DataFrame(results), use_container_width=True)


with tab4:
    st.header("Métricas")

    st.subheader("Learning to Rank")

    baseline = ranking_metrics.get("baseline", {})
    ranker = ranking_metrics.get("xgb_ranker", {})

    metrics_df = pd.DataFrame([
        {
            "model": "Global popularity baseline",
            "ndcg_at_10": baseline.get("ndcg_at_10"),
            "map_at_10": baseline.get("map_at_10"),
            "num_queries": baseline.get("num_queries"),
        },
        {
            "model": "XGBRanker",
            "ndcg_at_10": ranker.get("ndcg_at_10"),
            "map_at_10": ranker.get("map_at_10"),
            "num_queries": ranker.get("num_queries"),
        }
    ])

    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Embedding latency")

    st.json(embedding_latency)

    st.subheader("Negative sampling")

    st.json(ranking_metrics.get("negative_sampling", {}))