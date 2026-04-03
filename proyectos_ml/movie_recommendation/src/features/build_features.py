import pandas as pd
import os
import json
from typing import List, Optional
from pathlib import Path
import numpy as np

PATH_BASE = Path(__file__).resolve().parents[2]
INTERIM_DIR = PATH_BASE / "data" / "interim"
PROCESSED_DIR = PATH_BASE / "data" / "processed"

def load_interim(file_name: str, var: str=None) -> pd.DataFrame:
    path = INTERIM_DIR / file_name
    ext = path.suffix.lower()
    if ext == ".csv":
        if var is not None:
            return pd.read_csv(path, converters={var: lambda x: json.loads(x) if pd.notna(x) else None})
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Formato del archivo no soportado: {ext}")

def movies_with_genres(df: pd.DataFrame, var: str) -> pd.DataFrame:
    df = df.copy()
    df_genres = df[df[var].apply(lambda x: "(no genres listed)" not in x)]
    return df_genres

def save_processed(df: pd.DataFrame, file_name: str, var: str = None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / file_name
    ext = path.suffix.lower()
    if ext == ".csv":
        if var is not None:
            df = df.copy()
            df[var] = df[var].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
        df.to_csv(path, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
         raise ValueError(f"Formato del archivo no soportado: {ext}")

if __name__ == "__main__":
    pass