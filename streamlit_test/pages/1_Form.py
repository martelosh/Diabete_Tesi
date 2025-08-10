# streamlit_test/pages/1_Form.py
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from src.utils import load_best_model, predict_with_model, preprocess_for_inference  # noqa: E402

st.set_page_config(page_title="Form rischio diabete", page_icon="📝", layout="centered")
st.title("📝 Form di autovalutazione")

for k in ["pending_record","pending_model_type","pending_model_artifact","pending_pred_class"]:
    st.session_state.setdefault(k, None)

try:
    model, model_type, meta = load_best_model()
    st.sidebar.caption(f"Usando: **{model_type}**")
except Exception as e:
    st.error(e); st.stop()

def _inputs():
    s = lambda label: st.selectbox(label, [0, 1])
    sl = lambda label, a, b, d: st.slider(label, a, b, d)
    gender = s("Sesso (0=femmina, 1=maschio)"); age = sl("Età", 18, 90, 40)
    highbp=s("Hai la pressione alta?"); highchol=s("Hai il colesterolo alto?"); cholcheck=s("Hai controllato il colesterolo (5 anni)?")
    smoker=s("Fumi?"); stroke=s("Hai avuto un ictus?"); heartdisease=s("Hai malattie cardiache?"); physactivity=s("Fai attività fisica?")
    fruits=s("Mangi frutta regolarmente?"); veggies=s("Mangi verdura regolarmente?"); hvyalcoh=s("Bevi molto alcol?")
    anyhealthcare=s("Hai accesso a servizi sanitari?"); nomedicalcare=s("Hai evitato cure per costi?")
    genhlth=sl("Salute generale (1=Ottima, 5=Pessima)",1,5,3); menthlth=sl("Giorni problemi mentali (30)",0,30,2); physhlth=sl("Giorni problemi fisici (30)",0,30,2)
    diffwalk=s("Hai difficoltà a camminare?"); education=sl("Istruzione (1–6)",1,6,4); income=sl("Reddito (1–8)",1,8,4)
    peso=st.number_input("Peso (kg)",30.0,250.0,70.0,0.5); h=st.number_input("Altezza (cm)",100.0,220.0,170.0,0.5)
    bmi=(peso/((h/100)**2)) if h>0 else 0.0; st.write(f"👉 BMI: **{bmi:.2f}**")
    return pd.DataFrame([{
        "HighBP": int(highbp), "HighChol": int(highchol), "CholCheck": int(cholcheck), "BMI": round(float(bmi), 1),
        "Smoker": int(smoker), "Stroke": int(stroke), "HeartDiseaseorAttack": int(heartdisease), "PhysActivity": int(physactivity),
        "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(hvyalcoh), "AnyHealthcare": int(anyhealthcare),
        "NoDocbcCost": int(nomedicalcare), "GenHlth": int(genhlth), "MentHlth": int(menthlth), "PhysHlth": int(physhlth),
        "DiffWalk": int(diffwalk), "Sex": int(gender), "Age": int(age), "Education": int(education), "Income": int(income)
    }])

if st.button("🧪 Calcola predizione"):
    df_rec = _inputs()
    X = preprocess_for_inference(df_rec, meta)
    try:
        pred_class = int(predict_with_model(model, model_type, X)[0])
    except Exception as e:
        st.error(e); st.stop()
    st.session_state.update(pending_record=df_rec, pending_model_type=model_type,
                            pending_model_artifact=("best_keras_model.keras" if (PROJECT_ROOT/"data/grid_search_results/best_keras_model.keras").exists()
                                                    else ("best_keras_model.h5" if model_type=="keras" else
                                                          (sorted((PROJECT_ROOT/"data/grid_search_results").glob("*_optimized_model.pkl"),
                                                                  key=lambda p: p.stat().st_mtime, reverse=True)[0].name if model_type=="sklearn" else "unknown.pkl"))),
                            pending_pred_class=pred_class)
    st.success(f"Predizione: **{pred_class}** (0=No, 1=Pre, 2=Diabete)")

if st.session_state["pending_record"] is not None and st.session_state["pending_pred_class"] is not None:
    pred = st.session_state["pending_pred_class"]
    ok = st.radio("Risultato corretto?", ["Sì","No"], horizontal=True, index=0)
    label = pred if ok=="Sì" else st.selectbox("Valore corretto:", [0,1,2], index=pred)
    if st.button("💾 Salva con feedback"):
        out = st.session_state["pending_record"].copy()
        out["Predicted"]=pred; out["Diabetes_012"]=int(label)
        out["timestamp"]=datetime.now(timezone.utc).isoformat()
        out["model_type"]=st.session_state["pending_model_type"]
        out["model_artifact"]=st.session_state["pending_model_artifact"]
        p = PROJECT_ROOT / "data" / "training_feedback.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, mode="a", header=not p.exists(), index=False)
        st.success("✅ Salvato in data/training_feedback.csv")
        for k in ["pending_record","pending_model_type","pending_model_artifact","pending_pred_class"]:
            st.session_state[k]=None