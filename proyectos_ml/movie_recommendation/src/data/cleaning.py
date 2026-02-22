import pandas as pd
import numpy as np
import kagglehub
import re
import os
import json

RAW_DIR = "../../data/raw"
INTERIM_DIR = "../../data/interim"
PROCESSED_DIR = "../../data/processed"

def load_from_github(repo:str, file_path:str) -> pd.DataFrame:
    url = f"https://raw.githubusercontent.com/{repo}/master/{file_path}"
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(url)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(url)
    raise ValueError(f"Formato del archivo no soportado: {ext}")

def load_from_kaggle(dataset: str, file_name:str) -> pd.DataFrame:
    path_kaggle = kagglehub.dataset_download(dataset, file_name)
    ext = os.path.splitext(file_name)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path_kaggle)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path_kaggle)
    raise ValueError(f"Formato del archivo no soportado: {ext}")

def load_raw_data(file_name:str, subdir:str | None = None, **read_kwargs) -> pd.DataFrame:
    base = os.path.join(RAW_DIR, subdir) if subdir else RAW_DIR
    path = os.path.join(base, file_name)
    ext = os.path.splitext(file_name)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path, **read_kwargs)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, **read_kwargs)
    raise ValueError(f"Formato del archivo no soportado: {ext}")

def basic_clean(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["genres"] = df["genres"].apply(lambda x: x.split("|"))
    df["title"] = df["title"].apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", x))
    return df

# extrayendo el procentaje correspondiente de cada puntaje de rating para cada película
def extract_rating_percentages(df:pd.DataFrame) -> pd.DataFrame:
    rating_percentajes = []
    for rating in df["rating"].unique():
        group = df.loc[df["rating"]==rating]
        if len(group) >= 1000:
            proportion = len(group)/len(df)
            rating_percentajes.append(group.sample(frac=proportion))
        else:
            print(f"Rating {rating} contiene menos de 1000 registros, se asignará un porcentaje de 0")
            proportion = 0
    return pd.concat(rating_percentajes).reset_index(drop=True)

def save_interim(df: pd.DataFrame, filename: str, var: str = None) -> str:
    os.makedirs(INTERIM_DIR, exist_ok=True)
    path = os.path.join(INTERIM_DIR, filename)
    ext = os.path.splitext(filename)[-1].lower()
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
    return path

if __name__ == "__main__":

    ratings_df = load_from_github("jeknov/movieRec/refs/heads", "ratings.csv")
    print(f"Datos cargados desde GitHub ratings_df: {ratings_df.shape}")
    print(ratings_df.head())
    movies_df = load_from_github("jeknov/movieRec/refs/heads", "movies.csv")
    print(f"Datos cargados desde GitHub movies_df: {movies_df.shape}")
    print(movies_df.head())

    movies_clean_df = basic_clean(movies_df)
    print(movies_clean_df.head())
    ratings_filter_df = extract_rating_percentages(ratings_df)
    print(f"Datos filtrados por rating: {ratings_filter_df.shape}")

    save_interim(movies_clean_df, "movies_clean.csv", "genres")
    save_interim(ratings_filter_df, "ratings_filtered.csv")