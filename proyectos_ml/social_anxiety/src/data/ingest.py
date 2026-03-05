import pandas as pd
import numpy as np
import os

RAW_DIR = "../../data/raw"
INTERIM_DIR = "../../data/interim"

def load_raw_data(file_name: str, sub_dir:str | None = None, **kwargs) -> pd.DataFrame:
    base = os.path.join(RAW_DIR, sub_dir) if sub_dir else RAW_DIR
    path = os.path.join(base, file_name)
    ext = os.path.splitext(file_name)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path, **kwargs)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Formato del archivo no soportado: {ext}")

if __name__ == "__main__":
    pass