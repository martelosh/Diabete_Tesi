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

# ==== FILE FEEDBACK: PERCORSO FORZATO (NESSUNA AMBIGUITÀ) ====
# >>> Sostituisci il path qui sotto SOLO se la tua cartella del progetto è in un percorso diverso.
FEEDBACK = Path(r"C:\Users\MARTINADICORATO\Diabete_New\Diabete\data\training_feedback.csv").resolve()
DATA_DIR = FEEDBACK.parent
METRICS_DIR = DATA_DIR / "metrics"

st.caption(f"📂 Feedback path usato: {FEEDBACK}")
st.caption(f"📁 Metrics dir: {METRICS_DIR}")
st.caption(f"📌 Working dir dell'app: {Path.cwd()}")
st.caption(f"Esiste feedback? {'✅ sì' if FEEDBACK.exists() else '❌ no'}")

# Crea file vuoto con le colonne giuste se non esiste
if not FEEDBACK.exists():
    if st.button("Crea file feedback (vuoto)"):
        FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
            "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth",
            "DiffWalk","Sex","Age","Education","Income","Predicted","Diabetes_012","timestamp","model_type","model_artifact"
        ]).to_csv(FEEDBACK, index=False)
        st.success(f"Creato: {FEEDBACK}")

# Pulsante per scaricare il file effettivamente usato
if FEEDBACK.exists():
    st.download_button("⬇️ Scarica training_feedback.csv",
                       data=FEEDBACK.read_bytes(),
                       file_name="training_feedback.csv",
                       mime="text/csv")

# Strumento rapido: aggiungi una riga di test per verificare la scrittura
with st.expander("🧰 Strumenti rapidi"):
    if st.button("Aggiungi riga di test (debug)"):
        FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        test_row = pd.DataFrame([{
            "HighBP":0,"HighChol":0,"CholCheck":1,"BMI":22.2,"Smoker":0,"Stroke":0,"HeartDiseaseorAttack":0,"PhysActivity":1,
            "Fruits":0,"Veggies":0,"HvyAlcoholConsump":0,"AnyHealthcare":1,"NoDocbcCost":0,"GenHlth":3,"MentHlth":0,"PhysHlth":0,
            "DiffWalk":0,"Sex":0,"Age":30,"Education":4,"Income":4,
            "Predicted":0,"Diabetes_012":0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type":"debug","model_artifact":"debug"
        }])
        if FEEDBACK.exists() and FEEDBACK.stat().st_size > 0:
            old = pd.read_csv(FEEDBACK)
            new = pd.concat([old, test_row], ignore_index=True)
        else:
            new = test_row
        new.to_csv(FEEDBACK, index=False)
        st.success(f"Riga di test aggiunta a: {FEEDBACK}")

st.write("---")

# ==== UI SEMPLICE (HOME + FORM) ====
st.title("🧪 Rischio Diabete — Ambiente TEST")
st.caption("Ambiente di prova per validare il flusso di predizione e il salvataggio del feedback.")

# Carica modello
model, model_type, meta = None, None, {}
try:
    model, model_type, meta = load_best_model()
    st.success(f"Modello caricato: **{model_type}**")
except Exception as e:
    st.warning(f"Modello non caricato: {e}")

st.subheader("📋 Form di autovalutazione")

# --- Helper per record dal form ---
def _build_record(**vals) -> pd.DataFrame:
    bmi = vals["peso"] / max((vals["altezza_cm"]/100.0)**2, 1e-6)
    return pd.DataFrame([{
        "HighBP":int(vals["highbp"]), "HighChol":int(vals["highchol"]), "CholCheck":int(vals["cholcheck"]),
        "BMI":round(float(bmi),1), "Smoker":int(vals["smoker"]), "Stroke":int(vals["stroke"]),
        "HeartDiseaseorAttack":int(vals["heartdisease"]), "PhysActivity":int(vals["physactivity"]),
        "Fruits":int(vals["fruits"]), "Veggies":int(vals["veggies"]), "HvyAlcoholConsump":int(vals["hvyalcoh"]),
        "AnyHealthcare":int(vals["anyhealthcare"]), "NoDocbcCost":int(vals["nomedicalcare"]),
        "GenHlth":int(vals["genhlth"]), "MentHlth":int(vals["menthlth"]), "PhysHlth":int(vals["physhlth"]),
        "DiffWalk":int(vals["diffwalk"]), "Sex":int(vals["gender"]), "Age":int(vals["age"]),
        "Education":int(vals["education"]), "Income":int(vals["income"]),
    }])

# --- Form inputs ---
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
with c1: peso = st.number_input("Peso (kg)", 30.0, 250.0, 70.0, 0.5)
with c2: altezza_cm = st.number_input("Altezza (cm)", 100.0, 220.0, 170.0, 0.5)
bmi = peso / max((altezza_cm/100.0)**2, 1e-6)
st.caption(f"👉 BMI: **{bmi:.2f}**")

# --- Predizione + feedback ---
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
            st.session_state["pending_record"] = rec
            st.session_state["pending_pred_class"] = pred_class
            st.session_state["pending_model_type"] = model_type

            # nome artefatto (best sklearn o keras)
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

# --- Conferma feedback e salvataggio ---
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

            if FEEDBACK.exists() and FEEDBACK.stat().st_size > 0:
                df_old = pd.read_csv(FEEDBACK)
                df_new = pd.concat([df_old, out], ignore_index=True)
            else:
                df_new = out

            df_new.to_csv(FEEDBACK, index=False)
            # pulizia stato
            for k in ["pending_record","pending_pred_class","pending_model_type","pending_model_artifact"]:
                st.session_state[k] = None

            st.success(f"✅ Salvato in: {FEEDBACK}")
        except Exception as e:
            st.error(f"Errore nel salvataggio: {e}")

# === Debug opzionale ===
with st.expander("🔍 Debug"):
    st.write("Model type:", model_type)
    st.write("Meta:", meta)
    if st.session_state.get("pending_record") is not None:
        st.write("Input record:")
        st.dataframe(st.session_state["pending_record"], use_container_width=True)
        try:
            X_dbg = preprocess_for_inference(st.session_state["pending_record"], meta)
            st.write("X preprocessato:")
            st.dataframe(X_dbg, use_container_width=True)
        except Exception as e:
            st.write("Errore preprocess debug:", e)
