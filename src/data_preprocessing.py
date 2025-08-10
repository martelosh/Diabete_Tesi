# src/data_preprocessing.py
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_db_engine(username: str, password: str, host: str, port: int, database: str):
    url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    return create_engine(url)

def test_connection(engine):
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT NOW();"))
            print("Connessione riuscita:", list(result)[0][0])
    except Exception as e:
        print("Errore nella connessione:", e)

def load_data_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame, target_column: str = "Diabetes_012"):
    """
    - dropna
    - NON tocca la target
    - scala solo colonne numeriche 'continue' (euristica: nunique>13)
    Ritorna: df_clean, scaler, scaled_cols
    """
    df_clean = df.copy()

    # Assicurati che la target sia int
    if target_column in df_clean.columns:
        df_clean[target_column] = df_clean[target_column].astype("int")

    # Drop NA basilare (eventualmente sostituisci con imputazione)
    df_clean = df_clean.dropna().reset_index(drop=True)

    # Colonne numeriche candidate (escludi target)
    num_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != target_column]

    # Heuristica: continua se molti valori distinti
    continuous_cols = [c for c in num_cols if df_clean[c].nunique() > 13]

    # Fit-transform globale (NB: più corretto fittare SOLO su train; vedi nota sotto)
    scaler = StandardScaler()
    if continuous_cols:
        df_clean[continuous_cols] = scaler.fit_transform(df_clean[continuous_cols])

    return df_clean, scaler, continuous_cols

def import_dataframe_to_db(df: pd.DataFrame, table_name: str, engine):
    df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
    print(f"Dati importati con successo nella tabella '{table_name}'")
