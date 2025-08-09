# streamlit/pages/1_Form.py
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# Path al progetto per importare src/utils.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # root del progetto
sys.path.append(str(PROJECT_ROOT / "src"))

from utils import load_best_model, predict_with_model, preprocess_for_inference  # noqa: E402

st.set_page_config(page_title="Form rischio diabete", page_icon="📝", layout="centered")
st.title("📝 Form di autovalutazione del rischio diabete")
st.markdown("Compila tutti i campi per simulare il tuo rischio.")

# ---------- Inizializza stato ----------
for k, v in {
    "pending_record": None,     # DataFrame con il record utente (features)
    "pending_meta": None,       # meta per preprocess
    "pending_model_type": None, # 'sklearn' | 'keras'
    "pending_model_artifact": None,
    "pending_pred_class": None, # int
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Carica modello (auto-selezione) ----------
try:
    model, model_type, meta = load_best_model()
    st.sidebar.caption(f"Selezione automatica → usando: **{model_type}**")
except Exception as e:
    st.error(f"Errore nel caricamento del modello: {e}")
    st.stop()

# ---------- Input utente ----------
gender = st.selectbox("Sesso (0=femmina, 1=maschio)", [0, 1])
age = st.slider("Età", 18, 90, 40)
highbp = st.selectbox("Hai la pressione alta?", [0, 1])
highchol = st.selectbox("Hai il colesterolo alto?", [0, 1])
cholcheck = st.selectbox("Hai controllato il colesterolo negli ultimi 5 anni?", [0, 1])
smoker = st.selectbox("Fumi?", [0, 1])
stroke = st.selectbox("Hai avuto un ictus?", [0, 1])
heartdisease = st.selectbox("Hai malattie cardiache?", [0, 1])
physactivity = st.selectbox("Fai attività fisica?", [0, 1])
fruits = st.selectbox("Mangi frutta regolarmente?", [0, 1])
veggies = st.selectbox("Mangi verdura regolarmente?", [0, 1])
hvyalcoh = st.selectbox("Bevi molto alcol?", [0, 1])
anyhealthcare = st.selectbox("Hai accesso a servizi sanitari?", [0, 1])
nomedicalcare = st.selectbox("Hai evitato cure per costi?", [0, 1])
genhlth = st.slider("Salute generale (1=Ottima, 5=Pessima)", 1, 5, 3)
menthlth = st.slider("Giorni con problemi mentali (ultimi 30)", 0, 30, 2)
physhlth = st.slider("Giorni con problemi fisici (ultimi 30)", 0, 30, 2)
diffwalk = st.selectbox("Hai difficoltà a camminare?", [0, 1])
education = st.slider("Istruzione (1=elementari, 6=laurea)", 1, 6, 4)
income = st.slider("Reddito (1=basso, 8=alto)", 1, 8, 4)

peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
altezza_cm = st.number_input("Altezza (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.5)
altezza_m = altezza_cm / 100
bmi = peso / (altezza_m**2) if altezza_m > 0 else 0.0
st.write(f"👉 Il tuo BMI calcolato è: **{bmi:.2f}**")

# ---------- Step 1: calcola predizione ----------
if st.button("🧪 Calcola predizione"):
    # costruisci record
    new_record = {
        "HighBP": int(highbp),
        "HighChol": int(highchol),
        "CholCheck": int(cholcheck),
        "BMI": round(float(bmi), 1),
        "Smoker": int(smoker),
        "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heartdisease),
        "PhysActivity": int(physactivity),
        "Fruits": int(fruits),
        "Veggies": int(veggies),
        "HvyAlcoholConsump": int(hvyalcoh),
        "AnyHealthcare": int(anyhealthcare),
        "NoDocbcCost": int(nomedicalcare),
        "GenHlth": int(genhlth),
        "MentHlth": int(menthlth),
        "PhysHlth": int(physhlth),
        "DiffWalk": int(diffwalk),
        "Sex": int(gender),
        "Age": int(age),
        "Education": int(education),
        "Income": int(income),
    }
    df_rec = pd.DataFrame([new_record])

    # preprocess + predict
    X = preprocess_for_inference(df_rec, meta)
    try:
        pred = predict_with_model(model, model_type, X)
        pred_class = int(pred[0])
    except Exception as e:
        st.error(f"Errore durante la predizione: {e}")
        st.stop()

    # salva in sessione per lo step 2
    st.session_state["pending_record"] = df_rec
    st.session_state["pending_meta"] = meta
    st.session_state["pending_model_type"] = model_type

    artifacts_dir = PROJECT_ROOT / "data" / "grid_search_results"
    if model_type == "sklearn":
        cand = list(artifacts_dir.glob("*_optimized_model.pkl"))
        model_artifact = cand[0].name if cand else "unknown.pkl"
    else:
        # supporto sia al nuovo formato .keras che allo storico .h5
        if (artifacts_dir / "best_keras_model.keras").exists():
            model_artifact = "best_keras_model.keras"
        else:
            model_artifact = "best_keras_model.h5"
    st.session_state["pending_model_artifact"] = model_artifact
    st.session_state["pending_pred_class"] = pred_class

    st.success(f"Predizione calcolata: **{pred_class}** (0=No diabete, 1=Pre, 2=Diabete)")
    st.info("Ora conferma il feedback qui sotto e premi **💾 Salva con feedback**.")

# ---------- Step 2: feedback + salvataggio ----------
if st.session_state["pending_record"] is not None and st.session_state["pending_pred_class"] is not None:
    pred_class = st.session_state["pending_pred_class"]
    feedback = st.radio("Questo risultato è corretto?", ["Sì", "No"], horizontal=True, index=0)
    label = pred_class if feedback == "Sì" else st.selectbox(
        "Inserisci il valore corretto:", [0, 1, 2], index=pred_class
    )

    if st.button("💾 Salva con feedback"):
        save_path = PROJECT_ROOT / "data" / "training_feedback.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        out = st.session_state["pending_record"].copy()
        out["Predicted"] = pred_class
        out["Diabetes_012"] = int(label)
        out["timestamp"] = datetime.now(timezone.utc).isoformat()
        out["model_type"] = st.session_state["pending_model_type"]
        out["model_artifact"] = st.session_state["pending_model_artifact"]

        if save_path.exists():
            df_old = pd.read_csv(save_path)
            df_new = pd.concat([df_old, out], ignore_index=True)
        else:
            df_new = out

        df_new.to_csv(save_path, index=False)
        st.success("✅ Dati salvati in 'data/training_feedback.csv'.")

        # pulizia stato
        st.session_state["pending_record"] = None
        st.session_state["pending_meta"] = None
        st.session_state["pending_model_type"] = None
        st.session_state["pending_model_artifact"] = None
        st.session_state["pending_pred_class"] = None
