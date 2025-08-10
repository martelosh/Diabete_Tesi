from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_db_engine(username: str, password: str, host: str, port: int, database: str):
    """Crea un engine SQLAlchemy (MySQL via pymysql)."""
    url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    return create_engine(url)

def test_connection(engine):
    """Esegue una query semplice per verificare la connessione."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT NOW();"))
            print("Connessione riuscita:", list(result)[0][0])
    except Exception as e:
        print("Errore nella connessione:", e)

def load_data_from_csv(path: str) -> pd.DataFrame:
    """Carica un CSV in DataFrame."""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame, target_column: str | None = "Diabetes_012") -> pd.DataFrame:
    """
    Pulizia base + scaling solo sulle colonne davvero continue.
    - dropna semplice
    - individua colonne numeriche (escludendo la target se indicata)
    - considera "continue" quelle con molti valori distinti (nunique > 13)
    - applica StandardScaler solo a quelle
    """
    df_clean = df.dropna().reset_index(drop=True)

    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in num_cols:
        num_cols.remove(target_column)

    continuous_cols = [c for c in num_cols if df_clean[c].nunique() > 13]

    if continuous_cols:
        scaler = StandardScaler()
        df_clean[continuous_cols] = scaler.fit_transform(df_clean[continuous_cols])

    return df_clean

def import_dataframe_to_db(df: pd.DataFrame, table_name: str, engine):
    """Scrive una tabella (replace)."""
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
    print(f"Dati importati con successo nella tabella '{table_name}'")

def append_dataframe_to_db(df: pd.DataFrame, table_name: str, engine):
    """Appende righe a una tabella esistente."""
    df.to_sql(name=table_name, con=engine, if_exists="append", index=False)
    print(f"Dati aggiunti alla tabella '{table_name}'")