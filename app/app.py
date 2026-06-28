import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Anime Recommender", layout="wide")
st.title("🎌 Anime Recommender System")

# ─── TAB LAYOUT ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🤖 AI Recommender (LLM)", "📊 Similarity Search (TF-IDF)"])


# ─── TAB 1 : LLM-based recommender (original anime_recommender) ─────────────
with tab1:
    st.subheader("Ask anything about anime — powered by Groq LLM + ChromaDB")

    @st.cache_resource
    def init_pipeline():
        return AnimeRecommendationPipeline()

    pipeline = init_pipeline()

    query = st.text_input("Enter anime name or describe what you want to watch:", key="llm_query")
    if query:
        with st.spinner("Fetching AI recommendation..."):
            try:
                res = pipeline.recommend(query)
                st.success("Done!")
                st.write(res)
            except Exception as e:
                st.error(f"Error: {e}")


# ─── TAB 2 : TF-IDF cosine similarity (from Anime-Recommendation-system) ────
with tab2:
    st.subheader("Find similar anime by name — powered by TF-IDF & Cosine Similarity")

    @st.cache_data
    def load_anime_data():
        """Load anime.csv — works with both dataset column formats."""
        try:
            df = pd.read_csv("data/anime.csv", encoding="utf-8", on_bad_lines="skip")
        except FileNotFoundError:
            st.error("anime.csv not found in data/ folder. Please add it.")
            return None

        # Support both dataset formats
        # Format A (Anime-Recommendation-system): name, genre, type, episodes, rating, members
        # Format B (anime_recommender): Name, Genres, sypnopsis
        if "name" in df.columns:
            df.rename(columns={"name": "Name", "genre": "Genres"}, inplace=True)
        if "Genres" not in df.columns and "genre" in df.columns:
            df["Genres"] = df["genre"]
        df = df.dropna(subset=["Name"])
        df["Genres"] = df["Genres"].fillna("Unknown")
        return df

    @st.cache_data
    def build_tfidf_matrix(df):
        tfidf = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(df["Genres"])
        return tfidf, matrix

    def get_similar_anime(anime_name: str, df: pd.DataFrame, matrix, top_n: int = 10):
        name_lower = anime_name.strip().lower()
        matches = df[df["Name"].str.lower() == name_lower]

        if matches.empty:
            # Partial match fallback
            matches = df[df["Name"].str.lower().str.contains(name_lower, na=False)]

        if matches.empty:
            return None, []

        idx = matches.index[0]
        matched_name = df.loc[idx, "Name"]

        sim_scores = cosine_similarity(matrix[idx], matrix).flatten()
        sim_scores[idx] = 0  # exclude itself
        top_indices = np.argsort(sim_scores)[::-1][:top_n]

        results = df.iloc[top_indices][["Name", "Genres"]].copy()
        results["Similarity Score"] = sim_scores[top_indices].round(3)
        results = results.reset_index(drop=True)
        results.index += 1
        return matched_name, results

    df = load_anime_data()

    if df is not None:
        _, tfidf_matrix = build_tfidf_matrix(df)

        col1, col2 = st.columns([3, 1])
        with col1:
            anime_input = st.text_input("Enter an anime name to find similar ones:", key="tfidf_query")
        with col2:
            top_n = st.slider("Number of results", min_value=5, max_value=20, value=10)

        if anime_input:
            matched_name, results = get_similar_anime(anime_input, df, tfidf_matrix, top_n)
            if matched_name is None:
                st.warning(f"No anime found matching '{anime_input}'. Try a different name.")
            else:
                st.success(f"Showing {len(results)} anime similar to **{matched_name}**")
                st.dataframe(results, use_container_width=True)
