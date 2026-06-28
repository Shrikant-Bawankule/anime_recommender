import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv


load_dotenv()
st.set_page_config(page_title="ANime Recomender", layout="wide")


@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()


pipeline = init_pipeline()

st.title("Anime Recmmender System")

query = st.text_input("Enter the Anime name :")
if query:
    with st.spinner("Fetching the recommendation"):
        res = pipeline.recommend(query)
        st.success("Done!")
        st.write(res)
