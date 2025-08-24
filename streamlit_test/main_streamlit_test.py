# streamlit_test/main_streamlit_test.py
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st

# ==== PATH PROGETTO / IMPORT ====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_best_model, predict_with_model, preprocess_for_inference  # noqa: E402

# ==== CONFIG APP ====
st.set_page_config(page_title="Rischio Diabete — TEST", page_icon="🧪", layout="wide")

# ==== PATH DATI ====
DATA_DIR = PROJECT_ROOT / "data"
FEEDBACK = DATA_DIR / "feedback_test.csv"   # <— nuovo file (prima: training_feedback.csv)
METRICS_DIR = DATA_DIR / "metrics"

# -----------------------------------------------------------------------------
# NAVIGAZIONE
# -----------------------------------------------------------------------------
if "nav" not in st.session_state:
    st.session_state.nav = "🏠 Home"

# Gestisce eventuali redirect (dai bottoni) PRIMA di istanziare la radio
if st.session_state.get("go_to"):
    st.session_state.nav = st.session_state.pop("go_to")

nav = st.sidebar.radio(
    "Navigazione",
    ["🏠 Home", "📝 Compila form", "📊 Monitoraggio"],
    index=["🏠 Home", "📝 Compila form", "📊 Monitoraggio"].index(st.session_state.nav),
    key="nav",
)

# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
def page_home():
    st.title("🧪 Rischio Diabete — Ambiente TEST")
    st.caption("Benvenuta! Qui puoi compilare un form per ottenere una predizione e, in seguito, monitorare i feedback raccolti.")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("👉 Vai al form", use_container_width=True, type="primary"):
            st.session_state["go_to"] = "📝 Compila form"
            st.rerun()  # <— compatibilità: era st.experimental_rerun
    with c2:
        if st.button("📊 Vai al monitoraggio", use_container_width=True):
            st.session_state["go_to"] = "📊 Monitoraggio"
            st.rerun()  # <— compatibilità

    st.write("---")
    st.subheader("Come funziona")
    st.markdown(
        """
1. Compila il **form** con le tue informazioni: calcoliamo il BMI automaticamente.  
2. Visualizzi la **predizione** (0 = nessun rischio, 1 = pre-diabete, 2 = diabete).  
3. Confermi se il risultato è corretto (o lo correggi) e **salvi il feedback**.  
4. Nella pagina **Monitoraggio** vedi i riepiloghi e l’andamento dei feedback.
        """
    )

# -----------------------------------------------------------------------------
# FORM
# -----------------------------------------------------------------------------
def _build_record(**vals) -> pd.DataFrame:
    bmi = vals["peso"] / max((vals["altezza_cm"] / 100.0) ** 2, 1e-6)
    return pd.DataFrame([{
        "HighBP": int(vals["highbp"]),
        "HighChol": int(vals["highchol"]),
        "CholCheck": int(vals["cholcheck"]),
        "BMI": round(float(bmi), 1),
        "Smoker": int(vals["smoker"]),
        "Stroke": int(vals["stroke"]),
        "HeartDiseaseorAttack": int(vals["heartdisease"]),
        "PhysActivity": int(vals["physactivity"]),
        "Fruits": int(vals["fruits"]),
        "Veggies": int(vals["veggies"]),
        "HvyAlcoholConsump": int(vals["hvyalcoh"]),
        "AnyHealthcare": int(vals["anyhealthcare"]),
        "NoDocbcCost": int(vals["nomedicalcare"]),
        "GenHlth": int(vals["genhlth"]),
        "MentHlth": int(vals["menthlth"]),
        "PhysHlth": int(vals["physhlth"]),
        "DiffWalk": int(vals["diffwalk"]),
        "Sex": int(vals["gender"]),
        "Age": int(vals["age"]),
        "Education": int(vals["education"]),
        "Income": int(vals["income"]),
    }])

def page_form():
    st.title("📝 Compila form")
    # Carica modello
    model, model_type, meta = None, None, {}
    try:
        model, model_type, meta = load_best_model()
        st.success(f"Modello caricato: **{model_type}**")
    except Exception as e:
        st.warning(f"Modello non caricato: {e}")

    st.subheader("📋 Autovalutazione")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Sesso (0=femmina, 1=maschio)", [0, 1])
        age = st.slider("Età", 18, 90, 40)
        highbp = st.selectbox("Pressione alta?", [0, 1])
        highchol = st.selectbox("Colesterolo alto?", [0, 1])
        cholcheck = st.selectbox("Controllo colesterolo (ultimi 5 anni)?", [0, 1])
        smoker = st.selectbox("Fumi?", [0, 1])
        stroke = st.selectbox("Ictus in passato?", [0, 1])
        heartdisease = st.selectbox("Malattie cardiache?", [0, 1])
        physactivity = st.selectbox("Attività fisica regolare?", [0, 1])
        fruits = st.selectbox("Frutta regolare?", [0, 1])
    with col2:
        veggies = st.selectbox("Verdura regolare?", [0, 1])
        hvyalcoh = st.selectbox("Consumo elevato di alcol?", [0, 1])
        anyhealthcare = st.selectbox("Accesso a servizi sanitari?", [0, 1])
        nomedicalcare = st.selectbox("Eviti cure per costi?", [0, 1])
        genhlth = st.slider("Salute generale (1 ottima – 5 pessima)", 1, 5, 3)
        menthlth = st.slider("Giorni con problemi mentali (30 gg)", 0, 30, 2)
        physhlth = st.slider("Giorni con problemi fisici (30 gg)", 0, 30, 2)
        diffwalk = st.selectbox("Difficoltà a camminare?", [0, 1])
        education = st.slider("Istruzione (1–6)", 1, 6, 4)
        income = st.slider("Reddito (1–8)", 1, 8, 4)

    c1, c2 = st.columns(2)
    with c1:
        peso = st.number_input("Peso (kg)", 30.0, 250.0, 70.0, 0.5)
    with c2:
        altezza_cm = st.number_input("Altezza (cm)", 100.0, 220.0, 170.0, 0.5)

    bmi = peso / max((altezza_cm / 100.0) ** 2, 1e-6)
    st.caption(f"👉 BMI: **{bmi:.2f}**")

    # Predizione
    if st.button("🧪 Calcola predizione", type="primary", use_container_width=True):
        if model is None:
            st.error("Nessun modello caricato.")
        else:
            rec = _build_record(
                gender=gender, age=age, highbp=highbp, highchol=highchol, cholcheck=cholcheck,
                smoker=smoker, stroke=stroke, heartdisease=heartdisease, physactivity=physactivity,
                fruits=fruits, veggies=veggies, hvyalcoh=hvyalcoh, anyhealthcare=anyhealthcare,
                nomedicalcare=nomedicalcare, genhlth=genhlth, menthlth=menthlth, physhlth=physhlth,
                diffwalk=diffwalk, education=education, income=income, peso=peso, altezza_cm=altezza_cm
            )
            try:
                X = preprocess_for_inference(rec, meta)
                pred = predict_with_model(model, model_type, X)
                pred_class = int(pred[0])

                # Stato per feedback
                st.session_state["pending_record"] = rec
                st.session_state["pending_pred_class"] = pred_class
                st.session_state["pending_model_type"] = model_type

                # Nome artefatto
                artifacts_dir = PROJECT_ROOT / "data" / "grid_search_results"
                if model_type == "sklearn":
                    cand = list(artifacts_dir.glob("*_optimized_model.pkl"))
                    model_artifact = cand[0].name if cand else "unknown.pkl"
                else:
                    model_artifact = "best_keras_model.keras" if (artifacts_dir / "best_keras_model.keras").exists() else "best_keras_model.h5"
                st.session_state["pending_model_artifact"] = model_artifact

                st.success(f"Predizione: **{pred_class}**  (0=No, 1=Pre, 2=Diabete)")
            except Exception as e:
                st.error(f"Errore durante la predizione: {e}")

    # Feedback + salvataggio
    if st.session_state.get("pending_record") is not None:
        pred_class = st.session_state.get("pending_pred_class", 0)
        fb = st.radio("Questo risultato è corretto?", ["Sì", "No"], horizontal=True, index=0)
        label = pred_class if fb == "Sì" else st.selectbox("Valore corretto:", [0, 1, 2], index=pred_class)

        if st.button("💾 Salva con feedback", use_container_width=True):
            try:
                FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
                out = st.session_state["pending_record"].copy()
                out["Predicted"] = pred_class
                out["Diabetes_012"] = int(label)
                out["timestamp"] = datetime.now(timezone.utc).isoformat()
                out["model_type"] = st.session_state.get("pending_model_type", "sklearn")
                out["model_artifact"] = st.session_state.get("pending_model_artifact", "unknown.pkl")

                if FEEDBACK.exists():
                    df_old = pd.read_csv(FEEDBACK)
                    df_new = pd.concat([df_old, out], ignore_index=True)
                else:
                    df_new = out

                df_new.to_csv(FEEDBACK, index=False)

                # pulizia stato
                for k in ["pending_record", "pending_pred_class", "pending_model_type", "pending_model_artifact"]:
                    st.session_state[k] = None

                st.success("✅ Feedback salvato correttamente!")
                st.info("Vai alla pagina **Monitoraggio** per vedere i riepiloghi.")
            except Exception as e:
                st.error(f"Errore nel salvataggio: {e}")

# -----------------------------------------------------------------------------
# MONITORAGGIO
# -----------------------------------------------------------------------------
def page_monitor():
    st.title("📊 Monitoraggio")
    if not FEEDBACK.exists() or FEEDBACK.stat().st_size == 0:
        st.info("Non ci sono ancora feedback salvati.")
        return

    try:
        df = pd.read_csv(FEEDBACK)
    except Exception as e:
        st.error(f"Errore nella lettura del file di feedback: {e}")
        return

    # KPI rapidi
    st.subheader("KPI")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Totale feedback", len(df))
    with c2:
        st.metric("Modelli usati", df.get("model_type", pd.Series(dtype=str)).nunique())
    with c3:
        st.metric("Classi distinte", df.get("Diabetes_012", pd.Series(dtype=int)).nunique())

    # Distribuzione classi
    st.subheader("Distribuzione classi (Diabetes_012)")
    if "Diabetes_012" in df.columns:
        class_counts = df["Diabetes_012"].value_counts().sort_index()
        st.bar_chart(class_counts)
    else:
        st.write("Colonna 'Diabetes_012' non presente.")

    # Trend settimanale (se timestamp presente)
    st.subheader("Trend settimanale")
    if "timestamp" in df.columns:
        t = pd.to_datetime(df["timestamp"], errors="coerce")
        grp = pd.DataFrame({"ts": t}).dropna()
        if not grp.empty:
            grp["week"] = grp["ts"].dt.to_period("W").dt.start_time
            weekly = grp.groupby("week").size().rename("count")
            st.line_chart(weekly)
        else:
            st.write("Nessun timestamp valido per trend.")
    else:
        st.write("Colonna 'timestamp' non presente.")

    # Per modello
    st.subheader("Feedback per modello")
    if "model_artifact" in df.columns:
        per_model = df["model_artifact"].value_counts()
        st.bar_chart(per_model)
    else:
        st.write("Colonna 'model_artifact' non presente.")

    # Ultime righe
    st.subheader("Ultimi 10 feedback")
    st.dataframe(df.tail(10), use_container_width=True)

# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------
if nav == "🏠 Home":
    page_home()
elif nav == "📝 Compila form":
    page_form()
else:
    page_monitor()
