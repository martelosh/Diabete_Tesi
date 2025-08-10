from pathlib import Path
import pandas as pd

# Default percorsi (sovrascrivibili via argomenti delle funzioni)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_CSV = DATA_DIR / "training_feedback.csv"   # con label reale (Diabetes_012)
PRED_LOG_CSV = DATA_DIR / "prediction_log.csv"      # senza label (solo uso)

def build_weekly_feedback_report(feedback_csv: Path | str = FEEDBACK_CSV, save: bool = False):
    """
    Report settimanale su dati con LABEL (accuracy nel tempo) + confusion matrix.
    Ritorna (weekly_df, cm_df, by_model_df | None). Se save=True, salva i CSV in data/metrics/.
    """
    path = Path(feedback_csv)
    if not path.exists():
        raise FileNotFoundError(f"Non trovo {path}. Inserisci almeno un feedback.")

    df = pd.read_csv(path)

    required = {"timestamp", "Predicted", "Diabetes_012"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne: {', '.join(sorted(missing))}")

    # cast e pulizia
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["Predicted"] = pd.to_numeric(df["Predicted"], errors="coerce")
    df["Diabetes_012"] = pd.to_numeric(df["Diabetes_012"], errors="coerce")
    df = df.dropna(subset=["timestamp", "Predicted", "Diabetes_012"]).copy()
    df["Predicted"] = df["Predicted"].astype(int)
    df["Diabetes_012"] = df["Diabetes_012"].astype(int)

    # accuracy per riga
    df["is_correct"] = (df["Predicted"] == df["Diabetes_012"]).astype(int)

    # weekly (settimana che termina di lunedì)
    weekly = (
        df.set_index("timestamp")
          .groupby(pd.Grouper(freq="W-MON"))
          .agg(tests=("is_correct", "size"), accuracy=("is_correct", "mean"))
          .reset_index()
          .rename(columns={"timestamp": "week_start"})
          .sort_values("week_start")
    )
    weekly["accuracy"] = weekly["accuracy"].round(4)

    # confusion matrix complessiva
    cm = pd.crosstab(df["Diabetes_012"], df["Predicted"], rownames=["True"], colnames=["Pred"])

    # breakdown per modello se disponibile
    if "model_artifact" in df.columns:
        by_model = (
            df.groupby("model_artifact")
              .agg(tests=("is_correct", "size"), accuracy=("is_correct", "mean"))
              .reset_index()
        )
        by_model["accuracy"] = by_model["accuracy"].round(4)
    else:
        by_model = None

    if save:
        weekly.to_csv(METRICS_DIR / "weekly_report.csv", index=False)
        cm.to_csv(METRICS_DIR / "confusion_matrix_overall.csv")
        if by_model is not None:
            by_model.to_csv(METRICS_DIR / "by_model_report.csv", index=False)

    return weekly, cm, by_model


def build_usage_report(pred_log_csv: Path | str = PRED_LOG_CSV, save: bool = False):
    """
    Report settimanale di UTILIZZO (senza label): volume predizioni e distribuzione classi.
    Ritorna (weekly_counts_df, class_dist_df). Se save=True, salva i CSV in data/metrics/.
    """
    path = Path(pred_log_csv)
    if not path.exists():
        raise FileNotFoundError(f"Non trovo {path}. Popola prima il prediction log.")

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Nel prediction log manca la colonna 'timestamp'.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # conteggi settimanali
    weekly_counts = (
        df.set_index("timestamp")
          .groupby(pd.Grouper(freq="W-MON"))
          .size()
          .reset_index(name="predictions")
          .rename(columns={"timestamp": "week_start"})
          .sort_values("week_start")
    )

    # distribuzione classi predette (se presente 'Predicted' o 'predicted_class')
    pred_col = "Predicted" if "Predicted" in df.columns else ("predicted_class" if "predicted_class" in df.columns else None)
    if pred_col:
        class_dist = df[pred_col].value_counts(dropna=False).reset_index()
        class_dist.columns = ["class", "count"]
    else:
        class_dist = pd.DataFrame(columns=["class", "count"])

    if save:
        weekly_counts.to_csv(METRICS_DIR / "usage_weekly_counts.csv", index=False)
        class_dist.to_csv(METRICS_DIR / "usage_class_distribution.csv", index=False)

    return weekly_counts, class_dist