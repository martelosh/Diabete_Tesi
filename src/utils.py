# src/utils.py
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "grid_search_results"
META_PATH = ARTIFACTS_DIR / "model_meta.json"   # <-- contiene info su uso scaler & feature order

def _find_best_sklearn_model():
    if not ARTIFACTS_DIR.exists():
        return None
    cands = list(ARTIFACTS_DIR.glob("*_optimized_model.pkl"))
    return cands[0] if cands else None

def _find_best_keras_model():
    path = ARTIFACTS_DIR / "best_keras_model.h5"
    return path if path.exists() else None

def _load_meta():
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"use_scaler": False, "feature_order": None}

def load_best_model():
    """
    Carica il miglior modello in base alle metriche in model_meta.json:
    - se keras_score > sklearn_score -> KERAS
    - altrimenti -> SKLEARN
    Fallback: se manca un artefatto, usa l'altro disponibile.
    Ritorna (model, model_type, meta)
    """
    meta = _load_meta()
    skl_path = _find_best_sklearn_model()
    keras_path = _find_best_keras_model()

    skl_score = meta.get("sklearn_score")
    k_score   = meta.get("keras_score")

    # prova decisione "auto" solo se entrambe le metriche ci sono
    if skl_score is not None and k_score is not None:
        if k_score > skl_score and keras_path and keras_load_model is not None:
            return keras_load_model(keras_path), "keras", meta
        if skl_path and skl_path.exists():
            with open(skl_path, "rb") as f:
                return pickle.load(f), "sklearn", meta

    # altrimenti fallback ordinato
    if skl_path and skl_path.exists():
        with open(skl_path, "rb") as f:
            return pickle.load(f), "sklearn", meta
    if keras_path and keras_load_model is not None:
        return keras_load_model(keras_path), "keras", meta

    raise FileNotFoundError("Nessun modello trovato. Esegui prima src.main.")

def _load_scaler():
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None

def preprocess_for_inference(df_row: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Applica scaler solo se meta['use_scaler'] è True.
    Mantiene l'ordine colonne in meta['feature_order'] se presente.
    """
    df_row = df_row.copy()

    # allinea ordine colonne se registrato
    feat_order = meta.get("feature_order")
    if feat_order:
        df_row = df_row.reindex(columns=feat_order, fill_value=0)

    if meta.get("use_scaler", False):
        scaler = _load_scaler()
        if scaler is not None:
            X_scaled = scaler.transform(df_row)
            return pd.DataFrame(X_scaled, columns=df_row.columns)
    return df_row

def predict_with_model(model, model_type: str, X: pd.DataFrame):
    if model_type == "sklearn":
        y_pred = model.predict(X)
        return np.asarray(y_pred).astype(int)
    if model_type == "keras":
        probs = model.predict(X, verbose=0)
        return np.argmax(probs, axis=1).astype(int)
    raise ValueError(f"model_type sconosciuto: {model_type}")
