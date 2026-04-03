from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
from pathlib import Path
import pandas as pd
import json

PATH_BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = PATH_BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PATH_BASE / "data" / "processed"

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

def train_title_vectorizer(df, var):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df[var])
    dump(vectorizer, MODEL_DIR / "title_vectorizer.joblib")
    dump(tfidf_matrix, MODEL_DIR / "title_tfidf_matrix.joblib")
    return tfidf_matrix

if __name__ == "__main__":
    # LOAD PROCESSED DATA
    df = load_processed("movies_with_genres.csv", var="genres")
    # TRAIN VECTORIZER
    tfidf_matrix = train_title_vectorizer(df, var="title")
    print("Vectorizer entrenado y guardado en el directorio de modelos.")