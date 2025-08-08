import os
from pathlib import Path
from dotenv import load_dotenv
from model_training import split_data, evaluate_models_cross_validation, tune_keras_model
from data_preprocessing import create_db_engine, test_connection, load_data_from_csv, preprocess_data, import_dataframe_to_db
from grid_search import run_grid_search_and_save, param_grids

project_root = Path(__file__).resolve().parent.parent  # esce da src e va alla root
env_path = project_root / '.env'

load_dotenv(dotenv_path=env_path)
username = os.getenv("SQL_USERNAME")
password = os.getenv("SQL_PASSWORD")
host = os.getenv("SQL_HOST")
port = 3306
database = os.getenv("SQL_DATABASE")
    
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "data" / "diabete_data.csv"

engine = create_db_engine(username, password, host, port, database)
test_connection(engine)

df = load_data_from_csv(csv_path)
df_clean = preprocess_data(df)
import_dataframe_to_db(df_clean, table_name="diabetes_data", engine=engine)

x_train, x_test, y_train, y_test = split_data(df_clean, target_column="Diabetes_012")

results, best_estimator, best_model_name = evaluate_models_cross_validation(x_train, y_train)
run_grid_search_and_save(best_estimator, param_grids[best_model_name], x_train, y_train, model_name=best_model_name)
print("Risultati della cross-validazione:", results)

model, val_acc = tune_keras_model(x_train, y_train, x_test, y_test)
model_save = project_root / "data" / "grid_search_results"
model.save(model_save / "best_keras_model.h5")
print(f"Modello migliore: {model}, Accuratezza di validazione: {val_acc:.4f}")

