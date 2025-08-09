# src/retrain.py
import os
from pathlib import Path
import pandas as pd

from data_preprocessing import (
    load_data_from_csv, preprocess_data,
    create_db_engine, append_dataframe_to_db, test_connection
)
from model_training import split_data, evaluate_models_cross_validation, tune_keras_model
from grid_search import run_grid_search_and_save, param_grids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_ORIG = DATA_DIR / "diabete_data.csv"
CSV_FEEDBACK = DATA_DIR / "training_feedback.csv"
ARTIFACTS = DATA_DIR / "grid_search_results"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def load_and_merge_data() -> pd.DataFrame:
    df_orig = load_data_from_csv(str(CSV_ORIG))
    df_orig = preprocess_data(df_orig)

    if CSV_FEEDBACK.exists():
        df_fb = pd.read_csv(CSV_FEEDBACK)
        # Controllo colonne minime richieste:
        required_cols = set(df_orig.columns)
        missing = required_cols - set(df_fb.columns)
        if missing:
            raise ValueError(f"Nel feedback mancano colonne: {missing}")

        # Tieni solo le colonne dell’originale (ordine coerente)
        df_fb = df_fb[df_orig.columns]
        df_all = pd.concat([df_orig, df_fb], ignore_index=True)
    else:
        df_all = df_orig

    return df_all

def retrain_and_save(username: str, password: str, host: str, database: str, port: int = 3306):
    # 1) Merge dataset
    df_all = load_and_merge_data()

    # 2) Append su DB
    engine = create_db_engine(username, password, host, port, database)
    test_connection(engine)
    append_dataframe_to_db(df_all, table_name="diabetes_data", engine=engine)

    # 3) Train/val split
    x_train, x_test, y_train, y_test = split_data(df_all, target_column="Diabetes_012")

    # 4) Selezione modello migliore via CV
    results, best_estimator, best_model_name = evaluate_models_cross_validation(x_train, y_train)
    print("Cross-val:", results)

    # 5) Grid search sul best
    best_model, gs = run_grid_search_and_save(
        best_estimator, param_grids[best_model_name], x_train, y_train, model_name=best_model_name
    )

    # 6) (Opzionale) Keras tuning + salvataggio scaler/meta
    try:
        model_keras, val_acc = tune_keras_model(x_train, y_train, x_test, y_test, max_epochs=50)
        model_keras.save(ARTIFACTS / "best_keras_model.h5")
        print(f"Keras val acc: {val_acc:.4f}")
    except Exception as e:
        print("Keras tuning saltato/errore:", e)

    print("Retrain completato. Modelli salvati in:", ARTIFACTS)

if __name__ == "__main__":
    # Leggi credenziali da .env come fai in main.py (se vuoi)
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    import os
    retrain_and_save(
        username=os.getenv("SQL_USERNAME"),
        password=os.getenv("SQL_PASSWORD"),
        host=os.getenv("SQL_HOST"),
        database=os.getenv("SQL_DATABASE"),
        port=3306
    )
