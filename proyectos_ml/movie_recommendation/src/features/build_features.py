import pandas as pd
import os
import json
from typing import List, Optional

INTERIM_DIR = "../../data/interim"
PROCESSED_DIR = "../../data/processed"

def load_interim(file_name: str, var: str=None) -> pd.DataFrame:
    path = os.path.join(INTERIM_DIR, file_name)
    ext = os.path.splitext(file_name)[-1].lower()
    if ext == ".csv":
        if var is not None:
            return pd.read_csv(path, converters={var: json.loads})
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Formato del archivo no soportado: {ext}")

def movies_with_genres(df: pd.DataFrame, var: str) -> pd.DataFrame:
    df = df.copy()
    df_genres = df[df[var].apply(lambda x: "(no genres listed)" not in x)]
    return df_genres

def save_processed(df: pd.DataFrame, file_name: str, var: str = None) -> None:
    path = os.path.join(PROCESSED_DIR, file_name)
    ext = os.path.splitext(file_name)[-1].lower()
    if ext == ".csv":
        if var is not None:
            df[var] = df[var].apply(json.dumps)
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
         raise ValueError(f"Formato del archivo no soportado: {ext}")

if __name__ == "__main__":
    df_movies = load_interim("movies_clean.csv", var="genres")
    df_ratings = load_interim("ratings_filtered.csv")

    df_genres = movies_with_genres(df_movies, var="genres")
    save_processed(df_genres, "movies_with_genres.csv", var="genres")
