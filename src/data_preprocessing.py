import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data_from_csv(path: str) -> pd.DataFrame:
    """Carica i dati da un file CSV."""
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Esegue preprocessing di base: rimozione NA e normalizzazione."""
    df_clean = df.dropna()
    numeric_cols = df_clean.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    return df_clean
