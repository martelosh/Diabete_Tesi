# streamlit/pages/2_Monitoraggio.py
from pathlib import Path
import streamlit as st
import pandas as pd

# Vai alla root del progetto (da .../streamlit/pages risalgo di 2)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"
FEEDBACK = DATA_DIR / "training_feedback.csv"

st.set_page_config(page_title="Monitoraggio modello", page_icon="📈", layout="centered")
st.title("📈 Monitoraggio modello")

# Funzione che costruisce i report se mancano
def _build_reports():
    if not FEEDBACK.exists():
        return False, "Non trovo 'data/training_feedback.csv'. Fai almeno un invio dal Form."
    df = pd.read_csv(FEEDBACK)
    if df.empty:
        return False, "'training_feedback.csv' è vuoto. Fai almeno un invio dal Form."

    df["is_correct"] = (df["Predicted"] == df["Diabetes_012"]).astype(int)
    # timestamp può essere stringa ISO; parse
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Weekly report (se ho timestamp valido)
    if df["timestamp"].notna().any():
        weekly = (
            df.set_index("timestamp")
              .groupby(pd.Grouper(freq="W-MON"))
              .agg(tests=("is_correct", "size"),
                   accuracy=("is_correct", "mean"))
              .reset_index()
              .rename(columns={"timestamp": "week_start"})
        )
        weekly["accuracy"] = weekly["accuracy"].round(4)
        weekly.to_csv(METRICS_DIR / "weekly_report.csv", index=False)

    # by model
    if "model_artifact" in df.columns:
        by_model = (
            df.groupby("model_artifact")
              .agg(tests=("is_correct", "size"),
                   accuracy=("is_correct", "mean"))
              .reset_index()
        )
        by_model["accuracy"] = by_model["accuracy"].round(4)
        by_model.to_csv(METRICS_DIR / "by_model_report.csv", index=False)

    # confusion matrix complessiva
    cm = pd.crosstab(df["Diabetes_012"], df["Predicted"], rownames=["True"], colnames=["Pred"])
    cm.to_csv(METRICS_DIR / "confusion_matrix_overall.csv")

    return True, "Report generati/aggiornati."

# Bottone per rigenerare
if st.button("🔄 Aggiorna report"):
    ok, msg = _build_reports()
    st.success(msg) if ok else st.warning(msg)

# Se i file non ci sono, prova a costruirli ora
if not (METRICS_DIR / "weekly_report.csv").exists() \
   and not (METRICS_DIR / "by_model_report.csv").exists() \
   and not (METRICS_DIR / "confusion_matrix_overall.csv").exists():
    _build_reports()

weekly_path = METRICS_DIR / "weekly_report.csv"
by_model_path = METRICS_DIR / "by_model_report.csv"
cm_path = METRICS_DIR / "confusion_matrix_overall.csv"

# UI
if weekly_path.exists():
    st.subheader("Andamento settimanale")
    weekly = pd.read_csv(weekly_path, parse_dates=["week_start"])
    if not weekly.empty:
        st.line_chart(weekly.set_index("week_start")[["tests", "accuracy"]])
        last = weekly.sort_values("week_start").tail(1)
        c1, c2 = st.columns(2)
        c1.metric("Test ultima settimana", int(last["tests"].iloc[0]))
        c2.metric("Accuracy ultima settimana", f"{last['accuracy'].iloc[0]*100:.1f}%")
    else:
        st.info("Nessun dato settimanale ancora.")

if by_model_path.exists():
    st.subheader("Prestazioni per modello")
    st.dataframe(pd.read_csv(by_model_path))

if cm_path.exists():
    st.subheader("Confusion matrix (complessiva)")
    st.dataframe(pd.read_csv(cm_path, index_col=0))

if not weekly_path.exists() and not by_model_path.exists() and not cm_path.exists():
    st.info("Nessun report trovato. Fai almeno un invio dal Form e poi clicca **Aggiorna report**.")
