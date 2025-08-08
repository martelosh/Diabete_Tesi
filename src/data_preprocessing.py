from sqlalchemy import create_engine, text
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_db_engine(username: str, password: str, host: str, port: int, database: str):
    """Crea l'engine SQLAlchemy per la connessione al database."""
    url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    return create_engine(url)

def test_connection(engine):
    """Verifica che la connessione al database sia attiva."""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT NOW();"))
            print("Connessione riuscita:", list(result)[0][0])
    except Exception as e:
        print("Errore nella connessione:", e)

def load_data_from_csv(path: str) -> pd.DataFrame:
    """Carica i dati da un file CSV."""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Esegue preprocessing di base: rimozione NA e normalizzazione."""
    df_clean = df.dropna()
    numeric_cols = df_clean.select_dtypes(include='number').columns
    scaler = StandardScaler()
    for col in numeric_cols:
        if df_clean[col].nunique() > 10:
            df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    return df_clean

def import_dataframe_to_db(df: pd.DataFrame, table_name: str, engine):
    """Importa un DataFrame in una tabella del database (sovrascrive se esiste)."""
    df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
    print(f"Dati importati con successo nella tabella '{table_name}'")
