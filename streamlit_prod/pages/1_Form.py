import sys, numpy as np
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from utils import load_best_model, preprocess_for_inference          # noqa: E402
from chatbot import deepseek_chat                                    # noqa: E402
from prompt_chatbot import build_dynamic_system_prompt               # noqa: E402

st.set_page_config(page_title="Valutazione — Form", page_icon="📝", layout="centered")
st.markdown("""<style>
#floating-chat{position:fixed;right:24px;bottom:24px;z-index:9999}
#floating-chat button{border-radius:999px;padding:14px 16px;font-weight:800;border:1px solid rgba(0,0,0,.1);background:#fff}
.chat-panel{position:fixed;right:24px;bottom:90px;z-index:9998;width:min(440px,94vw);max-height:72vh;overflow:auto;background:#fff;
border:1px solid rgba(0,0,0,.10);border-radius:18px;box-shadow:0 22px 52px rgba(0,0,0,.22);padding:.9rem}
#chat-teaser{position:fixed;right:88px;bottom:110px;z-index:9999;max-width:min(380px,72vw);background:#fff;color:#111;
border:1px solid rgba(0,0,0,.12);border-radius:14px;box-shadow:0 16px 40px rgba(0,0,0,.18);padding:10px 12px}
#chat-teaser:after{content:"";position:absolute;right:-10px;bottom:14px;border:10px solid;border-color:transparent transparent transparent #fff}
.teaser-actions{display:flex;gap:8px;margin-top:6px;justify-content:flex-end}
</style>""", unsafe_allow_html=True)

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

# --- UI form ---
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
    })]

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
        sys_prompt = build_dynamic_system_prompt(cls, prob)
        first = deepseek_chat([], system_prompt=sys_prompt)
        if first.startswith("⚠️") or first.lower().startswith("errore"): first = sys_prompt.split("\n")[0]
        first += "\n\n**Per aiutarti a prenotare, scrivimi il tuo _comune_.**"
        st.session_state.update(messages=[{"role":"assistant","content":first}],
                                chat_open=False, teaser_message=first, show_teaser=True)

# --- CHAT (stesso floating del main) ---
def render_chat():
    st.markdown('<div id="floating-chat"><form><button type="submit" name="chat" value="toggle">💬</button></form></div>', unsafe_allow_html=True)
    qp = st.query_params
    if qp.get("chat") == "toggle":
        if st.session_state.last_pred is None:
            st.session_state.update(teaser_message="Per iniziare, compila il form.", show_teaser=True, chat_open=False)
        else:
            st.session_state.update(chat_open=not st.session_state.chat_open, show_teaser=False)
        st.query_params.clear()
    if qp.get("chat") == "open":
        st.session_state.update(chat_open=True, show_teaser=False) if st.session_state.last_pred is not None \
            else st.session_state.update(teaser_message="Prima compila il form.", show_teaser=True, chat_open=False)
        st.query_params.clear()
    if qp.get("teaser") == "close": st.session_state.show_teaser=False; st.query_params.clear()

    if st.session_state.show_teaser and not st.session_state.chat_open and st.session_state.teaser_message:
        text = st.session_state.teaser_message
        if len(text)>240: text = text[:235].rstrip()+"…"
        actions = ('<button type="submit" name="chat" value="open">Apri chat</button>'
                   '<button type="submit" name="teaser" value="close">Chiudi</button>') \
                  if st.session_state.last_pred is not None else \
                  ('<button type="submit" name="chat" value="toggle">Compila il form</button>'
                   '<button type="submit" name="teaser" value="close">Chiudi</button>')
        st.markdown(f'<div id="chat-teaser"><div>{text}</div><div class="teaser-actions"><form>{actions}</form></div></div>', unsafe_allow_html=True)

    if st.session_state.chat_open and st.session_state.last_pred is not None:
        st.markdown('<div class="chat-panel">**Assistente**  \n<small>Non sostituisce un consulto medico.</small><br/><br/>',
                    unsafe_allow_html=True)
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        if prompt := st.chat_input("Scrivi un messaggio…"):
            st.session_state.messages.append({"role":"user","content":prompt})
            # la risposta vera la diamo dal main (stessa logica), qui manteniamo il pannello
            st.session_state.messages.append({"role":"assistant","content":"Ricevuto! Vai alla Home per i contatti/locali o continua qui."})
            with st.chat_message("assistant"): st.markdown(st.session_state.messages[-1]["content"])
        st.markdown('</div>', unsafe_allow_html=True)

render_chat()