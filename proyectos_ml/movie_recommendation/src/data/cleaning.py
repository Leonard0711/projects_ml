import pandas as pd
import numpy as np
import kagglehub
import re
import os
import json
from pathlib import Path

PATH_BASE = Path(__file__).resolve().parents[2]
RAW_DIR = PATH_BASE / "data" / "raw"
INTERIM_DIR = PATH_BASE / "data" / "interim"

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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
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
    df["genres"] = df["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    df["title"] = df["title"].astype(str).replace("[^a-zA-Z0-9 ]", "", regex=True)
    return df

# extrayendo el procentaje correspondiente de cada puntaje de rating para cada película
def percentage_group_sampling(df:pd.DataFrame, min_samples: int) -> pd.DataFrame:
    rating_percentajes = []
    for rating in df["rating"].unique():
        group = df.loc[df["rating"]==rating]
        if len(group) >= min_samples:
            prop = len(group)/len(df)
            rating_percentajes.append(group.sample(frac=prop))
        else:
            print(f"Rating {rating} contiene menos de {min_samples} registros, se asignará un porcentaje de 0")
    if not rating_percentajes:
        raise ValueError(f"Ningún rating cumple con el mínimo de {min_samples} registros")
    return pd.concat(rating_percentajes, axis=0).reset_index(drop=True)

def save_interim(df: pd.DataFrame, filename: str, var: str = None) -> str:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DIR / filename
    ext = path.suffix.lower()
    if ext == ".csv":
            if var is not None:
                df[var] = df[var].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
                df.to_csv(path, index=False)
            else:
                df.to_csv(path, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Formato del archivo no soportado: {ext}")
    return path

if __name__ == "__main__":
    pass