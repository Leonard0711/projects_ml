from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import nltk
import pandas as pd
import numpy as np
import os
import json
import re

INTERIM_DIR = "../data/interim"
PROCESSED_DIR = "../data/processed"

def load_processed(file_name: str, var: str=None) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, file_name)
    ext = os.path.splitext(file_name)[-1].lower()
    if ext == ".csv":
        if var is not None:
            return pd.read_csv(path, converters={var: json.loads})
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
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    title_vectorizer = vectorizer.fit_transform(df[var])
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    query_title = vectorizer.transform([title])
    similarity = cosine_similarity(title_vectorizer, query_title).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = df.iloc[indices]
    return results

if __name__ == "__main__":
    df_movies = load_processed("movies_with_genres.csv", var="genres")
    df_movies["title"] = tokenize_titles(df_movies, var="title")
    title_query = "kill"
    similar_titles = similarity_titles(df_movies, var="title", title=title_query)
    print(similar_titles[["movieid", "title", "genres"]])
