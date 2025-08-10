# streamlit_test/pages/2_Monitoraggio.py
from pathlib import Path
import sys
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from metrics import build_weekly_report  # noqa: E402

st.set_page_config(page_title="Monitoraggio modello", page_icon="📈", layout="centered")
st.title("📈 Monitoraggio modello")

try:
    weekly, cm, by_model = build_weekly_report()
except Exception as e:
    st.warning(str(e)); st.stop()

st.subheader("Andamento settimanale")
if not weekly.empty:
    st.line_chart(weekly.set_index("week_start")[["tests", "accuracy"]])
    last = weekly.sort_values("week_start").tail(1)
    c1, c2 = st.columns(2)
    c1.metric("Test ultima settimana", int(last["tests"].iloc[0]))
    c2.metric("Accuracy ultima settimana", f"{last['accuracy'].iloc[0]*100:.1f}%")
else:
    st.info("Nessun dato settimanale ancora.")

st.subheader("Prestazioni per modello")
st.dataframe(by_model if by_model is not None else pd.DataFrame(), use_container_width=True)

st.subheader("Confusion matrix (complessiva)")
st.dataframe(cm, use_container_width=True)