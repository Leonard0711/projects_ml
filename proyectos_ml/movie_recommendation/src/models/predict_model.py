from src.data.cleaning import load_from_github, basic_clean, percentage_group_sampling, save_interim
from src.features.build_features import load_interim, movies_with_genres, save_processed
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import nltk
from pathlib import Path
from joblib import load
import pandas as pd
import numpy as np
import json
import re

PATH_BASE = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PATH_BASE / "data" / "processed"
MODELS_DIR = PATH_BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_processed(file_name: str, var: str=None) -> pd.DataFrame:
    path = PROCESSED_DIR / file_name
    ext = path.suffix.lower()
    if ext == ".csv":
        if var is not None:
            return pd.read_csv(path, converters={var: lambda x: json.loads(x) if pd.notna(x) else None})
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Formato del archivo no soportado: {ext}")

def tokenize_titles(df: pd.DataFrame, var: str) -> pd.DataFrame:
    df = df.copy()
    titles = df[var]
    tokens_titles = titles.apply(wordpunct_tokenize)
    stop_words = set(stopwords.words("english"))
    titles_filtered = tokens_titles.apply(
        lambda token_title: " ".join([word for word in token_title if word.lower() not in stop_words])
    )
    return titles_filtered

def similarity_titles(df: pd.DataFrame, var: str, title: str) -> pd.DataFrame:
    vectorizer = load(MODELS_DIR / "title_vectorizer.joblib")
    tfidf_matrix = load(MODELS_DIR / "title_tfidf_matrix.joblib")
    clean_title = re.sub("[^a-zA-Z0-9 ]", "", title)
    query_title = vectorizer.transform([clean_title])
    similarity = cosine_similarity(tfidf_matrix, query_title).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = df.iloc[indices]
    return results

if __name__ == "__main__":
    # LOAD AND CLEAN DATA
    ratings =  load_from_github("jeknov/movieRec/refs/heads", "ratings.csv")
    print(f"Datos cargados desde GitHub ratings: {ratings.shape}")
    movies = load_from_github("jeknov/movieRec/refs/heads", "movies.csv")
    print(f"Datos cargados desde GitHub movies: {movies.shape}")
    movies_clean_df = basic_clean(movies)
    ratings_filter_df = percentage_group_sampling(ratings, min_samples=1000)
    print(f"Datos filtrados por rating: {ratings_filter_df.shape}")
    save_interim(movies_clean_df, "movies_clean.csv", "genres")
    save_interim(ratings_filter_df, "ratings_filtered.csv")

    # BUILD FEATURES
    df_movies = load_interim("movies_clean.csv", var="genres")
    df_ratings = load_interim("ratings_filtered.csv")
    df_genres = movies_with_genres(df_movies, var="genres")
    save_processed(df_genres, "movies_with_genres.csv", var="genres") 

    # PREDICTION
    df_movies = load_processed("movies_with_genres.csv", var="genres")
    df_movies["title"] = tokenize_titles(df_movies, var="title")
    title_query = "dinosaurs"
    similar_titles = similarity_titles(df_movies, var="title", title=title_query)
    print(similar_titles[["movieid", "title", "genres"]])

