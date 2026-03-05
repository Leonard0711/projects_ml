import pandas as pd

RAW_DIR = "../../data/raw"
INTERIM_DIR = "../../data/interim"

def delet_null_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

if __name__ == "__main__":
    pass