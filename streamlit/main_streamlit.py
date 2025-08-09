# streamlit/main_streamlit.py
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# === PATH PROGETTO / IMPORT UTILS ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # root del progetto
sys.path.append(str(PROJECT_ROOT / "src"))
from utils import load_best_model, predict_with_model, preprocess_for_inference  # noqa: E402

# === CONFIG & STILE ===
st.set_page_config(page_title="Rischio Diabete — Demo", page_icon="🩺", layout="wide")
st.markdown("""
<style>
/* Nascondi sidebar e hamburger */
section[data-testid="stSidebar"] {display: none;}
header [data-testid="baseButton-headerNoPadding"] {visibility: hidden;}
div.block-container {padding-top: 2rem; padding-bottom: 2rem;}
/* Hero */
.hero {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(0, 102, 255, 0.08), transparent 60%),
              radial-gradient(1000px 500px at 90% 30%, rgba(255, 0, 128, 0.08), transparent 60%);
  border-radius: 24px; padding: 2.6rem; border: 1px solid rgba(0,0,0,0.05);
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.card {border-radius: 18px; padding: 1.2rem; border: 1px solid rgba(0,0,0,0.06); background: #fff;
       box-shadow: 0 6px 16px rgba(0,0,0,0.05);}
.btn {display:inline-block; padding: 14px 18px; border-radius: 14px; font-weight: 600;
      border:1px solid rgba(0,0,0,0.1); background: white; transition: all .15s ease;}
.btn:hover {transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,0,0,0.07);}
.small {font-size: 0.92rem; opacity: 0.9;}
.topbar {display:flex; justify-content:flex-start; margin-bottom: .5rem;}
</style>
""", unsafe_allow_html=True)

# === ROUTER STATO ===
if "view" not in st.session_state:
    st.session_state.view = "home"

def go(view: str):
    st.session_state.view = view

# === HOME ===
def render_home():
    st.markdown(
        """
        <div class="hero">
          <h1>🩺 Valutazione del Rischio Diabete</h1>
          <p class="small">
            Demo interattiva: inserisci poche informazioni sul tuo stile di vita e salute,
            <b>stimiamo il rischio (0, 1, 2)</b> con il modello migliore (Sklearn o Keras),
            e tracciamo le prestazioni nel tempo.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns([1.3, 1.1, 1.3])

    with c1:
        st.markdown("### 🎯 Obiettivo")
        st.markdown(
            "- Raccogliere dati anonimi tramite **Form**;\n"
            "- Effettuare una **predizione** con il modello addestrato;\n"
            "- Salvare feedback per **migliorare** il modello nel tempo."
        )
    with c2:
        st.markdown("### 📦 Modello")
        try:
            _, model_type, meta = load_best_model()
            st.markdown(f"**In uso (auto)**: `{model_type}`")
            sk = meta.get("sklearn_score"); ke = meta.get("keras_score")
            st.markdown(f"- Sklearn CV: **{sk:.4f}**" if sk is not None else "- Sklearn CV: _n/d_")
            st.markdown(f"- Keras Val: **{ke:.4f}**" if ke is not None else "- Keras Val: _n/d_")
        except Exception as e:
            st.warning(f"Modello non caricato: {e}")
    with c3:
        st.markdown("### 🔒 Privacy")
        st.markdown("I dati sono salvati localmente in `data/training_feedback.csv` per scopi di test.")

    st.write("")
    a, b, _ = st.columns([1, 1, 1])
    with a:
        if st.button("📝 Apri Form", use_container_width=True, type="primary"):
            go("form")
    with b:
        if st.button("📈 Apri Monitoraggio", use_container_width=True):
            go("monitor")

# === FORM (two-step + back in alto) ===
def render_form():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if st.button("⬅️ Torna alla Home"):
        go("home")
        st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## 📝 Form di autovalutazione")
    try:
        model, model_type, meta = load_best_model()
        st.caption(f"Selezione automatica → usando: **{model_type}**")
    except Exception as e:
        st.error(f"Errore nel caricamento del modello: {e}")
        return

    # Stato two-step
    for k in ["pending_record","pending_model_type","pending_model_artifact","pending_pred_class"]:
        if k not in st.session_state: st.session_state[k] = None

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
        hvyalcoh = st.selectbox("Molto alcol?", [0, 1])
        anyhealthcare = st.selectbox("Accesso a servizi sanitari?", [0, 1])
        nomedicalcare = st.selectbox("Evitate cure per costi?", [0, 1])
        genhlth = st.slider("Salute generale (1 ottima – 5 pessima)", 1, 5, 3)
        menthlth = st.slider("Giorni con problemi mentali (30 gg)", 0, 30, 2)
        physhlth = st.slider("Giorni con problemi fisici (30 gg)", 0, 30, 2)
        diffwalk = st.selectbox("Difficoltà a camminare?", [0, 1])
        education = st.slider("Istruzione (1–6)", 1, 6, 4)
        income = st.slider("Reddito (1–8)", 1, 8, 4)

    c1, c2 = st.columns(2)
    with c1:
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
    with c2:
        altezza_cm = st.number_input("Altezza (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.5)
    altezza_m = altezza_cm / 100
    bmi = peso / (altezza_m**2) if altezza_m > 0 else 0.0
    st.write(f"👉 Il tuo BMI calcolato è: **{bmi:.2f}**")

    if st.button("🧪 Calcola predizione", type="primary"):
        new_record = {
            "HighBP": int(highbp), "HighChol": int(highchol), "CholCheck": int(cholcheck),
            "BMI": round(float(bmi), 1), "Smoker": int(smoker), "Stroke": int(stroke),
            "HeartDiseaseorAttack": int(heartdisease), "PhysActivity": int(physactivity),
            "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(hvyalcoh),
            "AnyHealthcare": int(anyhealthcare), "NoDocbcCost": int(nomedicalcare),
            "GenHlth": int(genhlth), "MentHlth": int(menthlth), "PhysHlth": int(physhlth),
            "DiffWalk": int(diffwalk), "Sex": int(gender), "Age": int(age),
            "Education": int(education), "Income": int(income),
        }
        df_rec = pd.DataFrame([new_record])

        X = preprocess_for_inference(df_rec, meta)
        try:
            pred = predict_with_model(model, model_type, X)
            pred_class = int(pred[0])
        except Exception as e:
            st.error(f"Errore durante la predizione: {e}")
            return

        artifacts_dir = PROJECT_ROOT / "data" / "grid_search_results"
        if model_type == "sklearn":
            cand = list(artifacts_dir.glob("*_optimized_model.pkl"))
            model_artifact = cand[0].name if cand else "unknown.pkl"
        else:
            model_artifact = "best_keras_model.keras" if (artifacts_dir / "best_keras_model.keras").exists() else "best_keras_model.h5"

        st.session_state.pending_record = df_rec
        st.session_state.pending_model_type = model_type
        st.session_state.pending_model_artifact = model_artifact
        st.session_state.pending_pred_class = pred_class

        st.success(f"Predizione: **{pred_class}**  (0=No, 1=Pre, 2=Diabete)")
        st.info("Conferma il feedback e poi salva.")

    # Step 2: feedback + save
    if st.session_state.pending_record is not None:
        pred_class = st.session_state.pending_pred_class
        feedback = st.radio("Questo risultato è corretto?", ["Sì", "No"], horizontal=True, index=0)
        label = pred_class if feedback == "Sì" else st.selectbox("Valore corretto:", [0, 1, 2], index=pred_class)

        if st.button("💾 Salva con feedback"):
            save_path = PROJECT_ROOT / "data" / "training_feedback.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            out = st.session_state.pending_record.copy()
            out["Predicted"] = pred_class
            out["Diabetes_012"] = int(label)
            out["timestamp"] = datetime.now(timezone.utc).isoformat()
            out["model_type"] = st.session_state.pending_model_type
            out["model_artifact"] = st.session_state.pending_model_artifact

            if save_path.exists():
                df_old = pd.read_csv(save_path)
                df_new = pd.concat([df_old, out], ignore_index=True)
            else:
                df_new = out

            df_new.to_csv(save_path, index=False)
            st.success("✅ Dati salvati in 'data/training_feedback.csv'.")

            # reset
            st.session_state.pending_record = None
            st.session_state.pending_model_type = None
            st.session_state.pending_model_artifact = None
            st.session_state.pending_pred_class = None

# === MONITORAGGIO (back in alto + auto-build report) ===
def render_monitor():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if st.button("⬅️ Torna alla Home"):
        go("home")
        st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## 📈 Monitoraggio modello")

    DATA_DIR = PROJECT_ROOT / "data"
    METRICS_DIR = DATA_DIR / "metrics"
    FEEDBACK = DATA_DIR / "training_feedback.csv"

    def build_reports():
        if not FEEDBACK.exists():
            return False, "Non trovo 'data/training_feedback.csv'. Fai almeno un invio dal Form."
        df = pd.read_csv(FEEDBACK)
        if df.empty:
            return False, "'training_feedback.csv' è vuoto."

        df["is_correct"] = (df["Predicted"] == df["Diabetes_012"]).astype(int)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        METRICS_DIR.mkdir(parents=True, exist_ok=True)

        # Weekly
        if df["timestamp"].notna().any():
            weekly = (
                df.set_index("timestamp")
                  .groupby(pd.Grouper(freq="W-MON"))
                  .agg(tests=("is_correct", "size"), accuracy=("is_correct", "mean"))
                  .reset_index()
                  .rename(columns={"timestamp": "week_start"})
            )
            weekly["accuracy"] = weekly["accuracy"].round(4)
            weekly.to_csv(METRICS_DIR / "weekly_report.csv", index=False)

        # By model
        if "model_artifact" in df.columns:
            by_model = (
                df.groupby("model_artifact")
                  .agg(tests=("is_correct", "size"), accuracy=("is_correct", "mean"))
                  .reset_index()
            )
            by_model["accuracy"] = by_model["accuracy"].round(4)
            by_model.to_csv(METRICS_DIR / "by_model_report.csv", index=False)

        # Confusion
        cm = pd.crosstab(df["Diabetes_012"], df["Predicted"], rownames=["True"], colnames=["Pred"])
        cm.to_csv(METRICS_DIR / "confusion_matrix_overall.csv")
        return True, "Report generati/aggiornati."

    # Auto-build se mancano file
    weekly_path = METRICS_DIR / "weekly_report.csv"
    by_model_path = METRICS_DIR / "by_model_report.csv"
    cm_path = METRICS_DIR / "confusion_matrix_overall.csv"
    if not weekly_path.exists() and not by_model_path.exists() and not cm_path.exists():
        build_reports()

    if st.button("🔄 Aggiorna report"):
        ok, msg = build_reports()
        (st.success if ok else st.warning)(msg)

    # UI
    if weekly_path.exists():
        weekly = pd.read_csv(weekly_path, parse_dates=["week_start"])
        st.markdown("### Andamento settimanale")
        if not weekly.empty:
            st.line_chart(weekly.set_index("week_start")[["tests", "accuracy"]])
            last = weekly.sort_values("week_start").tail(1)
            c1, c2 = st.columns(2)
            c1.metric("Test ultima settimana", int(last["tests"].iloc[0]))
            c2.metric("Accuracy ultima settimana", f"{last['accuracy'].iloc[0]*100:.1f}%")
        else:
            st.info("Nessun dato settimanale ancora.")

    if by_model_path.exists():
        st.markdown("### Prestazioni per modello")
        st.dataframe(pd.read_csv(by_model_path), use_container_width=True)

    if cm_path.exists():
        st.markdown("### Confusion matrix (complessiva)")
        st.dataframe(pd.read_csv(cm_path, index_col=0), use_container_width=True)

# === ROUTING ===
if st.session_state.view == "home":
    render_home()
elif st.session_state.view == "form":
    render_form()
elif st.session_state.view == "monitor":
    render_monitor()
else:
    go("home")
