# streamlit_test/main_streamlit_test.py
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from src.utils import load_best_model, predict_with_model, preprocess_for_inference  # noqa: E402
from src.from_streamlit.metrics_report import build_weekly_report

st.set_page_config(page_title="Rischio Diabete — Demo", page_icon="🩺", layout="wide")
st.session_state.setdefault("view", "home")
go = lambda v: st.session_state.update(view=v)

def _pick_artifact(model_type: str) -> str:
    g = (PROJECT_ROOT / "data" / "grid_search_results")
    if model_type == "sklearn":
        files = sorted(g.glob("*_optimized_model.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0].name if files else "unknown.pkl"
    return "best_keras_model.keras" if (g / "best_keras_model.keras").exists() else "best_keras_model.h5"

def _inputs():
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
        menthlth = st.slider("Giorni problemi mentali (30 gg)", 0, 30, 2)
        physhlth = st.slider("Giorni problemi fisici (30 gg)", 0, 30, 2)
        diffwalk = st.selectbox("Difficoltà a camminare?", [0, 1])
        education = st.slider("Istruzione (1–6)", 1, 6, 4)
        income = st.slider("Reddito (1–8)", 1, 8, 4)
    c1, c2 = st.columns(2)
    with c1:
        peso = st.number_input("Peso (kg)", 30.0, 250.0, 70.0, 0.5)
    with c2:
        altezza_cm = st.number_input("Altezza (cm)", 100.0, 220.0, 170.0, 0.5)
    bmi = (peso / ((altezza_cm / 100) ** 2)) if altezza_cm > 0 else 0.0
    st.write(f"👉 BMI: **{bmi:.2f}**")
    rec = {
        "HighBP": int(highbp), "HighChol": int(highchol), "CholCheck": int(cholcheck),
        "BMI": round(float(bmi), 1), "Smoker": int(smoker), "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heartdisease), "PhysActivity": int(physactivity),
        "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(hvyalcoh),
        "AnyHealthcare": int(anyhealthcare), "NoDocbcCost": int(nomedicalcare),
        "GenHlth": int(genhlth), "MentHlth": int(menthlth), "PhysHlth": int(physhlth),
        "DiffWalk": int(diffwalk), "Sex": int(gender), "Age": int(age),
        "Education": int(education), "Income": int(income),
    }
    return pd.DataFrame([rec])

def render_home():
    st.title("🩺 Valutazione del Rischio Diabete — Demo")
    try:
        _, t, m = load_best_model()
        st.write(f"In uso: **{t}**")
        sk, ke = m.get("sklearn_score"), m.get("keras_score")
        st.write(f"- Sklearn CV: **{sk:.4f}**" if sk is not None else "- Sklearn CV: _n/d_")
        st.write(f"- Keras Val: **{ke:.4f}**" if ke is not None else "- Keras Val: _n/d_")
    except Exception as e:
        st.warning(f"Modello non caricato: {e}")
    c1, c2 = st.columns(2)
    c1.button("📝 Apri Form", use_container_width=True, on_click=lambda: go("form"))
    c2.button("📈 Apri Monitoraggio", use_container_width=True, on_click=lambda: go("monitor"))

def render_form():
    st.button("⬅️ Home", on_click=lambda: go("home"))
    st.header("📝 Form di autovalutazione")
    try:
        model, model_type, meta = load_best_model()
        st.caption(f"Usando: **{model_type}**")
    except Exception as e:
        st.error(e); return

    for k in ["pending_record","pending_model_type","pending_model_artifact","pending_pred_class"]:
        st.session_state.setdefault(k, None)

    df_rec = _inputs()
    if st.button("🧪 Calcola predizione", type="primary"):
        X = preprocess_for_inference(df_rec, meta)
        try:
            pred_class = int(predict_with_model(model, model_type, X)[0])
        except Exception as e:
            st.error(e); return
        st.session_state.update(
            pending_record=df_rec,
            pending_model_type=model_type,
            pending_model_artifact=_pick_artifact(model_type),
            pending_pred_class=pred_class,
        )
        st.success(f"Predizione: **{pred_class}**  (0=No, 1=Pre, 2=Diabete)")
        st.info("Conferma/Correggi e salva.")

    if st.session_state.pending_record is not None:
        pred = st.session_state.pending_pred_class
        ok = st.radio("Risultato corretto?", ["Sì", "No"], horizontal=True, index=0)
        label = pred if ok == "Sì" else st.selectbox("Valore corretto:", [0, 1, 2], index=pred)
        if st.button("💾 Salva con feedback"):
            out = st.session_state.pending_record.copy()
            out["Predicted"] = pred
            out["Diabetes_012"] = int(label)
            out["timestamp"] = datetime.now(timezone.utc).isoformat()
            out["model_type"] = st.session_state.pending_model_type
            out["model_artifact"] = st.session_state.pending_model_artifact
            p = PROJECT_ROOT / "data" / "training_feedback.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(p, mode="a", header=not p.exists(), index=False)
            st.success("✅ Salvato in data/training_feedback.csv")
            for k in ["pending_record","pending_model_type","pending_model_artifact","pending_pred_class"]:
                st.session_state[k] = None

def render_monitor():
    st.button("⬅️ Home", on_click=lambda: go("home"))
    st.header("📈 Monitoraggio")
    try:
        weekly, cm, by_model = build_weekly_report()
    except Exception as e:
        st.warning(str(e)); return
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
    st.subheader("Confusion matrix complessiva")
    st.dataframe(cm, use_container_width=True)

{"home": render_home, "form": render_form, "monitor": render_monitor}[st.session_state.view]()