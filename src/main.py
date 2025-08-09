# src/main.py
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.model_training import split_data, evaluate_models_cross_validation, tune_keras_model
from src.data_preprocessing import create_db_engine, test_connection, preprocess_data
from src.grid_search import run_grid_search_and_save, param_grids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)
username = os.getenv("SQL_USERNAME")
password = os.getenv("SQL_PASSWORD")
host = os.getenv("SQL_HOST")
database = os.getenv("SQL_DATABASE")
port = int(os.getenv("SQL_PORT", 3306))

# 1) Connessione DB
engine = create_db_engine(username, password, host, port, database)
test_connection(engine)

# 2) Carico dati direttamente dal DB (tabella già esistente)
TABLE_NAME = "diabetes_data"
df = pd.read_sql_table(TABLE_NAME, con=engine)

# 3) Preprocessing leggero (dropna)
df_clean = preprocess_data(df)

# 4) Split e training
x_train, x_test, y_train, y_test = split_data(df_clean, target_column="Diabetes_012")

# 5) Cross-validation su più modelli
results, best_estimator, best_model_name = evaluate_models_cross_validation(x_train, y_train)
print("Risultati CV:", results)

# 6) Grid search sul migliore e salvataggio modello
best_model, gs = run_grid_search_and_save(
    best_estimator,
    param_grids[best_model_name],
    x_train, y_train,
    model_name=best_model_name
)

# 7) (Opzionale) Keras tuning con scaler salvato automaticamente
try:
    model_keras, val_acc = tune_keras_model(x_train, y_train, x_test, y_test, max_epochs=50)
    artifacts = PROJECT_ROOT / "data" / "grid_search_results"
    artifacts.mkdir(parents=True, exist_ok=True)
    model_keras.save(artifacts / "best_keras_model.h5")
    print(f"Keras val acc: {val_acc:.4f}")

    meta_path = artifacts / "model_meta.json"
    meta = {}
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["keras_score"] = float(val_acc)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
except Exception as e:
    print("Keras non eseguito:", e)
