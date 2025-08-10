import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

#python -m src.mainfrom src.model_training import split_data, evaluate_models_cross_validation, tune_keras_model
from src.model_training import split_data, evaluate_models_cross_validation, train_keras_simple
from src.data_preprocessing import create_db_engine, test_connection
from src.grid_search import run_grid_search_and_save as run_grid_search, param_grids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# 0) ENV / DB
load_dotenv(ENV_PATH)
username = os.getenv("SQL_USERNAME")
password = os.getenv("SQL_PASSWORD")
host     = os.getenv("SQL_HOST")
database = os.getenv("SQL_DATABASE")
port     = int(os.getenv("SQL_PORT", 3306))

# 1) Connessione DB
engine = create_db_engine(username, password, host, port, database)
test_connection(engine)

# 2) Carico dati già PULITI e SCALATI dal DB
TABLE_NAME = "diabetes_data"
df = pd.read_sql_table(TABLE_NAME, con=engine)

# 3) Split (STRATIFICATO) in train/test
x_train, x_test, y_train, y_test = split_data(df, target_column="Diabetes_012")

# 4) Cross-validation su più modelli (scelta algoritmo migliore)
results, best_estimator, best_model_name = evaluate_models_cross_validation(x_train, y_train)
print("Risultati CV (accuracy media):", results)
print("Miglior modello (CV):", best_model_name)

# 5) Grid search SOLO sul vincitore (nessun salvataggio qui)
#    Uso un'istanza “pulita” della stessa classe del best_estimator
estimator_cls = best_estimator.__class__
estimator_fresh = estimator_cls()
best_model, gs = run_grid_search(
    estimator=estimator_fresh,
    param_grid=param_grids[best_model_name],
    x_train=x_train, y_train=y_train,
    model_name=best_model_name,   # ← obbligatorio per la versione che salva
    cv=5, scoring="accuracy", verbose=0
)

# Ricavo ciò che ti serve dal GridSearchCV
best_params = gs.best_params_
best_cv = float(gs.best_score_)
cv_results = gs.cv_results_

print(f"GridSearch > {best_model_name} best params:", best_params)
print(f"GridSearch > {best_model_name} mean CV acc: {best_cv:.4f}")

# 6) (Opzionale) Keras: validation set interno, niente scaling qui
try:
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    model_keras, val_acc = train_keras_simple(x_tr, y_tr, x_val, y_val, max_epochs=50)
    print(f"Keras val acc: {val_acc:.4f}")
except Exception as e:
    print("Keras non eseguito:", e)