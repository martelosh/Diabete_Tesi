import sys, numpy as np
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from utils import load_best_model, preprocess_for_inference  # noqa: E402

st.set_page_config(page_title="Valutazione — Form", page_icon="📝", layout="centered")

# stato condiviso
for k, v in {"last_pred":None, "last_prob":None, "messages":[], "chat_open":False,
             "teaser_message":None, "show_teaser":False}.items():
    st.session_state.setdefault(k, v)

def predict_with_proba(model, model_type: str, X: pd.DataFrame):
    if model_type == "sklearn":
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]; c = int(np.argmax(p)); return c, float(p[c])
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            if s.ndim == 1:
                p1 = 1/(1+np.exp(-s[0])); c = int(p1 >= .5); return c, float(p1 if c else 1-p1)
            p = np.exp(s[0]-np.max(s[0])); p/=p.sum(); c = int(np.argmax(p)); return c, float(p[c])
        return int(model.predict(X)[0]), 0.50
    p = model.predict(X, verbose=0)[0]; c = int(np.argmax(p)); return c, float(p[c])

def make_teaser(pred: int, prob: float) -> str:
    label = {0: "nessun diabete", 1: "pre-diabete", 2: "diabete"}.get(int(pred), "")
    return (
        f"Risultato stimato: **{pred}** ({label}) — probabilità **{prob*100:.1f}%**.\n"
        "Questo non sostituisce un parere medico.\n\n"
        "Scrivimi il tuo **comune** per mostrarti i contatti utili e aiutarti a prenotare."
    )

# UI
left = st.columns([1,3])[0]
if left.button("⬅️ Torna alla Home"):
    try: st.switch_page("main_prod.py")
    except Exception: st.stop()
st.title("📋 Valutazione personale")
st.caption("Compila i campi. Il BMI è calcolato automaticamente.")

def _inputs_df():
    s=lambda t: st.selectbox(t,[0,1]); sl=lambda t,a,b,d: st.slider(t,a,b,d)
    c1,c2=st.columns(2)
    with c1:
        gender=s("Sesso (0=femmina, 1=maschio)"); age=sl("Età",18,90,40); highbp=s("Pressione alta?"); highchol=s("Colesterolo alto?")
        cholcheck=s("Controllo colesterolo (5 anni)?"); smoker=s("Fumi?"); stroke=s("Ictus in passato?"); heartdisease=s("Malattie cardiache?")
        physactivity=s("Attività fisica regolare?"); fruits=s("Frutta regolare?")
    with c2:
        veggies=s("Verdura regolare?"); hvyalcoh=s("Consumo elevato di alcol?"); anyhealthcare=s("Accesso a servizi sanitari?")
        nomedicalcare=s("Eviti cure per costi?"); genhlth=sl("Salute generale (1 ottima – 5 pessima)",1,5,3)
        menthlth=sl("Giorni problemi mentali (30)",0,30,2); physhlth=sl("Giorni problemi fisici (30)",0,30,2)
        diffwalk=s("Difficoltà a camminare?"); education=sl("Istruzione (1–6)",1,6,4); income=sl("Reddito (1–8)",1,8,4)
    b1,b2=st.columns(2)
    with b1: peso=st.number_input("Peso (kg)",30.0,250.0,70.0,0.5)
    with b2: h=st.number_input("Altezza (cm)",100.0,220.0,170.0,0.5)
    bmi=peso/((max(h,.0001)/100)**2); st.caption(f"BMI: {bmi:.2f}")
    return pd.DataFrame([{
        "HighBP":int(highbp),"HighChol":int(highchol),"CholCheck":int(cholcheck),"BMI":round(float(bmi),1),
        "Smoker":int(smoker),"Stroke":int(stroke),"HeartDiseaseorAttack":int(heartdisease),"PhysActivity":int(physactivity),
        "Fruits":int(fruits),"Veggies":int(veggies),"HvyAlcoholConsump":int(hvyalcoh),"AnyHealthcare":int(anyhealthcare),
        "NoDocbcCost":int(nomedicalcare),"GenHlth":int(genhlth),"MentHlth":int(menthlth),"PhysHlth":int(physhlth),
        "DiffWalk":int(diffwalk),"Sex":int(gender),"Age":int(age),"Education":int(education),"Income":int(income),
    }])

df = _inputs_df()
st.write("")
if st.button("🔎 Calcola risultato", type="primary"):
    try:
        model, model_type, meta = load_best_model()
        X = preprocess_for_inference(df, meta)
        cls, prob = predict_with_proba(model, model_type, X)
    except Exception as e:
        st.error(f"Errore durante la predizione: {e}")
    else:
        st.session_state.update(last_pred=cls, last_prob=prob)
        st.success(f"Risultato: **{cls}**  (0=No, 1=Pre, 2=Diabete)")
        st.info(f"Probabilità: **{prob*100:.1f}%**")
        first = make_teaser(cls, prob)
        st.session_state.update(messages=[{"role":"assistant","content":first}],
                                chat_open=False, teaser_message=first, show_teaser=True)
        st.caption("Apri la chat in basso a destra per consigli/prenotazioni.")