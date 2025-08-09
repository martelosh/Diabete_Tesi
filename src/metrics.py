# src/metrics.py
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEEDBACK = DATA_DIR / "training_feedback.csv"
METRICS_DIR = DATA_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def build_weekly_report():
    if not FEEDBACK.exists():
        raise FileNotFoundError("Non trovo data/training_feedback.csv. Fai almeno un test dal form.")

    df = pd.read_csv(FEEDBACK)
    if "timestamp" not in df.columns:
        raise ValueError("Nel CSV manca la colonna 'timestamp' (aggiungila nel salvataggio Streamlit).")

    # accuracy = match tra Predicted e Diabetes_012
    df["is_correct"] = (df["Predicted"] == df["Diabetes_012"]).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # raggruppo per settimana (Lunedì come start, va bene per IT)
    weekly = (
        df.set_index("timestamp")
          .groupby(pd.Grouper(freq="W-MON"))
          .agg(
              tests=("is_correct", "size"),
              accuracy=("is_correct", "mean")
          )
          .reset_index()
          .rename(columns={"timestamp": "week_start"})
    )
    weekly["accuracy"] = weekly["accuracy"].round(4)

    # salviamo report principale
    out = METRICS_DIR / "weekly_report.csv"
    weekly.to_csv(out, index=False)

    # confusion matrix complessiva (utile per capire gli errori)
    cm = pd.crosstab(df["Diabetes_012"], df["Predicted"], rownames=["True"], colnames=["Pred"])
    cm.to_csv(METRICS_DIR / "confusion_matrix_overall.csv")

    # breakdown per modello (opzionale ma utile)
    by_model = (
        df.groupby("model_artifact")
          .agg(tests=("is_correct", "size"), accuracy=("is_correct", "mean"))
          .reset_index()
    )
    by_model["accuracy"] = by_model["accuracy"].round(4)
    by_model.to_csv(METRICS_DIR / "by_model_report.csv", index=False)

    print(f"Salvato: {out}")
    print(f"Salvato: {METRICS_DIR / 'confusion_matrix_overall.csv'}")
    print(f"Salvato: {METRICS_DIR / 'by_model_report.csv'}")

if __name__ == "__main__":
    build_weekly_report()
