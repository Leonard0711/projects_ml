import os
import pandas as pd
import json
from sqlalchemy import create_engine
from typing import Optional
from pathlib import Path

password = os.getenv("MYSQL_PASSWORD")
engine_mysql = create_engine(f"mysql+pymysql://root:{password}@127.0.0.1:3306/TERCERA")

PATH_BASE = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PATH_BASE / "data" / "processed"

def query_file(name_query: str, folder: Optional[str] = "utils") -> str:
    path = PATH_BASE / "sql" / folder / f"{name_query}.sql"
    if not path.exists():
        raise FileNotFoundError(f"Archivo de consulta no encontrado: {path}")
    return path.read_text(encoding="utf-8")

def make_query(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    df = pd.read_sql(query, engine_mysql, params=params)
    return df

def extract_data(name_query: str, params: Optional[dict] = None) -> pd.DataFrame:
    query = query_file(name_query)
    df = make_query(query, params)
    return df

def save_extracted_data(df: pd.DataFrame, file_name:str, var: Optional[str]=None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / file_name
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

if __name__ == "__main__":
    pass