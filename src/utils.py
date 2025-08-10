import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Keras è opzionale: se non c'è, saltiamo in modo sicuro
try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "grid_search_results"
META_PATH = ARTIFACTS_DIR / "model_meta.json"   # contiene info su feature_order e punteggi

def _find_best_sklearn_model():
    """Restituisce il path del modello sklearn salvato (se presente)."""
    if not ARTIFACTS_DIR.exists():
        return None
    cands = list(ARTIFACTS_DIR.glob("*_optimized_model.pkl"))
    return cands[0] if cands else None  # semplice: prende il primo trovato

def _find_best_keras_model():
    """Restituisce il path del modello Keras salvato (se presente)."""
    path = ARTIFACTS_DIR / "best_keras_model.h5"
    return path if path.exists() else None

def _load_meta():
    """Carica le meta-info; se assenti, default minimale (solo feature_order=None)."""
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"feature_order": None}

def load_best_model():
    """
    Carica il modello migliore in base a model_meta.json:
    - se keras_score > sklearn_score -> KERAS
    - altrimenti -> SKLEARN
    Fallback: se manca un artefatto, usa l'altro disponibile.
    Ritorna (model, model_type, meta).
    """
    meta = _load_meta()
    skl_path = _find_best_sklearn_model()
    keras_path = _find_best_keras_model()

    skl_score = meta.get("sklearn_score")
    k_score   = meta.get("keras_score")

    # decisione automatica solo se entrambe le metriche sono presenti
    if skl_score is not None and k_score is not None:
        if k_score > skl_score and keras_path and keras_load_model is not None:
            return keras_load_model(keras_path), "keras", meta
        if skl_path and skl_path.exists():
            with open(skl_path, "rb") as f:
                return pickle.load(f), "sklearn", meta

    # fallback ordinato
    if skl_path and skl_path.exists():
        with open(skl_path, "rb") as f:
            return pickle.load(f), "sklearn", meta
    if keras_path and keras_load_model is not None:
        return keras_load_model(keras_path), "keras", meta

    raise FileNotFoundError("Nessun modello trovato. Esegui prima il training.")

def preprocess_for_inference(df_row: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Mantiene solo l'ordine colonne in meta['feature_order'] (se presente).
    Nessuna scalatura: i dati devono arrivare già preprocessati.
    """
    df_row = df_row.copy()
    feat_order = meta.get("feature_order")
    if feat_order:
        df_row = df_row.reindex(columns=feat_order, fill_value=0)
    return df_row

def predict_with_model(model, model_type: str, X: pd.DataFrame):
    """Wrapper di predizione: sklearn -> classi; keras -> argmax su softmax."""
    if model_type == "sklearn":
        y_pred = model.predict(X)
        return np.asarray(y_pred).astype(int)
    if model_type == "keras":
        probs = model.predict(X, verbose=0)
        return np.argmax(probs, axis=1).astype(int)
    raise ValueError(f"model_type sconosciuto: {model_type}")