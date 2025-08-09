# streamlit_prod/main_prod.py
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# ========== PATH & IMPORT ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from utils import load_best_model, preprocess_for_inference  # noqa: E402
from chatbot import deepseek_chat, get_nearby_contacts      # noqa: E402
from prompt_chatbot import build_dynamic_system_prompt      # noqa: E402

# ========== PAGE CONFIG & THEME ==========
st.set_page_config(page_title="Valutazione Rischio Diabete", page_icon="🩺", layout="wide")

st.markdown("""
<style>
/* —— Layout base —— */
section[data-testid="stSidebar"] { display: none; }
header [data-testid="baseButton-headerNoPadding"] { visibility: hidden; }
div.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* —— Hero —— */
.hero {
  border-radius: 28px;
  padding: clamp(1.6rem, 2.5vw, 2.6rem);
  border: 1px solid rgba(0,0,0,0.06);
  background:
    radial-gradient(1200px 600px at 8% 10%, rgba(0, 120, 255, .10), transparent 60%),
    radial-gradient(1000px 500px at 90% 30%, rgba(255, 60, 140, .10), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,255,255,0.88));
  box-shadow: 0 14px 44px rgba(0,0,0,.08);
}
.hero h1 { margin: 0 0 .6rem 0; }
.small { font-size: .95rem; opacity: .9; }

/* —— Cards & badges —— */
.card {
  border: 1px solid rgba(0,0,0,.06);
  background: #fff;
  border-radius: 18px;
  box-shadow: 0 8px 22px rgba(0,0,0,.06);
  padding: 1.1rem 1.2rem;
}
.badge {
  display:inline-block; padding:.28rem .6rem; border-radius:999px;
  border:1px solid rgba(0,0,0,.1); font-size:.82rem; background:#fff;
}

/* —— CTA buttons —— */
.stButton > button[kind="primary"] {
  border-radius: 14px; font-weight: 700; padding: .7rem 1rem;
  box-shadow: 0 10px 24px rgba(0,0,0,.10);
}
.stButton > button:not([kind="primary"]) {
  border-radius: 12px; font-weight: 600;
}

/* —— Floating chat button —— */
#floating-chat { position: fixed; right: 24px; bottom: 24px; z-index: 9999; }
#floating-chat button {
  border-radius: 999px; padding: 14px 16px; font-weight: 800;
  border: 1px solid rgba(0,0,0,.1); background: #fff;
  box-shadow: 0 14px 30px rgba(0,0,0,.15);
  cursor: pointer; transition: transform .12s ease, box-shadow .12s ease;
}
#floating-chat button:hover { transform: translateY(-2px); box-shadow: 0 16px 34px rgba(0,0,0,.18); }

/* —— Chat panel —— */
.chat-panel {
  position: fixed; right: 24px; bottom: 90px; z-index: 9998;
  width: min(440px, 94vw); max-height: 72vh; overflow: auto;
  background: #fff; border: 1px solid rgba(0,0,0,.10);
  border-radius: 18px; box-shadow: 0 22px 52px rgba(0,0,0,.22);
  padding: .9rem;
}

/* —— Teaser bubble —— */
#chat-teaser {
  position: fixed; right: 88px; bottom: 110px; z-index: 9999;
  max-width: min(380px, 72vw);
  background: #ffffff; color: #111;
  border: 1px solid rgba(0,0,0,.12);
  border-radius: 14px; box-shadow: 0 16px 40px rgba(0,0,0,.18);
  padding: 10px 12px; font-size: 0.95rem; line-height: 1.35;
  animation: teaserIn .18s ease-out;
}
#chat-teaser:after {
  content: ""; position: absolute; right: -10px; bottom: 14px;
  border-width: 10px; border-style: solid;
  border-color: transparent transparent transparent #ffffff;
  filter: drop-shadow(0 2px 2px rgba(0,0,0,.12));
}
#chat-teaser .teaser-actions { display: flex; gap: 8px; margin-top: 6px; justify-content: flex-end; }
#chat-teaser button {
  border: 1px solid rgba(0,0,0,.12); background: #fff; color: #000; border-radius: 10px;
  padding: 4px 8px; font-size: 0.85rem; cursor: pointer;
  transition: transform .12s ease, box-shadow .12s ease;
}
#chat-teaser button:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(0,0,0,.12); }

@keyframes teaserIn { from {opacity:0; transform:translateY(6px);} to {opacity:1; transform:translateY(0);} }

/* —— Dark mode —— */
@media (prefers-color-scheme: dark) {
  .hero { border-color: rgba(255,255,255,.08); background: linear-gradient(180deg,#0f1117,#0f121a); color:#e9e9ea; }
  .card, #floating-chat button, .chat-panel, #chat-teaser {
    background: #101318; color: #e9e9ea; border-color: rgba(255,255,255,.08);
  }
  #chat-teaser:after { border-color: transparent transparent transparent #101318; }
}
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if "view" not in st.session_state:
    st.session_state.view = "home"
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None         # 0/1/2
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None         # 0..1
if "messages" not in st.session_state:
    st.session_state.messages = []            # [{role, content}]
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "teaser_message" not in st.session_state:
    st.session_state.teaser_message = None
if "show_teaser" not in st.session_state:
    st.session_state.show_teaser = False

def go(view: str):
    st.session_state.view = view

# ========== PREDICT (classe + probabilità) ==========
def predict_with_proba(model, model_type: str, X: pd.DataFrame):
    import numpy as np
    if model_type == "sklearn":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            cls = int(probs[0].argmax())
            return cls, float(probs[0, cls])
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            if scores.ndim == 1:
                p1 = 1 / (1 + np.exp(-scores[0]))
                cls = int(p1 >= 0.5)
                return cls, float(p1 if cls == 1 else 1 - p1)
            exps = np.exp(scores[0] - np.max(scores[0]))
            probs = exps / np.sum(exps)
            cls = int(probs.argmax())
            return cls, float(probs[cls])
        cls = int(model.predict(X)[0])
        return cls, 0.50
    elif model_type == "keras":
        probs = model.predict(X, verbose=0)
        cls = int(probs[0].argmax())
        return cls, float(probs[0, cls])
    else:
        raise ValueError(f"model_type sconosciuto: {model_type}")

# ========== HOME ==========
def render_home():
    st.markdown(
        """
        <div class="hero">
          <h1>🩺 Valutazione del Rischio Diabete</h1>
          <p class="small">
            Inserisci poche informazioni sul tuo stato di salute e stile di vita: stimiamo il rischio
            (<b>0</b>=nessun diabete, <b>1</b>=pre-diabete, <b>2</b>=diabete) e mostriamo la
            <b>probabilità</b> associata. Un assistente ti aiuta per prenotazioni e consigli pratici.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2, col3 = st.columns([1.1, 1, 1.2])
    with col1:
        st.markdown("### 🎯 Perché usarlo")
        st.markdown(
            "- Screening rapido e informativo\n"
            "- Probabilità della predizione, non solo una classe\n"
            "- Supporto alla prenotazione visite tramite chat"
        )
    with col2:
        st.markdown("### 🔒 Privacy")
        st.markdown(
            "I dati sono usati solo per questa valutazione. \n"
            "Il risultato **non sostituisce** un consulto medico."
        )
    with col3:
        st.markdown("### ⚙️ Come funziona")
        st.markdown(
            "1) Compila il form\n"
            "2) Ricevi risultato + probabilità\n"
            "3) Apri la chat per prenotare un controllo"
        )
    st.write("")
    if st.button("📝 Apri il form", type="primary", use_container_width=True):
        go("form")

# ========== FORM (senza feedback; mostra probabilità) ==========
def render_form():
    top = st.columns([1, 3])[0]
    with top:
        if st.button("⬅️ Torna alla Home"):
            go("home"); st.stop()

    st.markdown("## 📋 Valutazione personale")
    st.caption("Compila i campi. Il BMI viene calcolato automaticamente.")

    c1, c2 = st.columns(2)
    with c1:
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
    with c2:
        veggies = st.selectbox("Verdura regolare?", [0, 1])
        hvyalcoh = st.selectbox("Consumo elevato di alcol?", [0, 1])
        anyhealthcare = st.selectbox("Accesso a servizi sanitari?", [0, 1])
        nomedicalcare = st.selectbox("Eviti cure per costi?", [0, 1])
        genhlth = st.slider("Salute generale (1 ottima – 5 pessima)", 1, 5, 3)
        menthlth = st.slider("Giorni con problemi mentali (ultimi 30)", 0, 30, 2)
        physhlth = st.slider("Giorni con problemi fisici (ultimi 30)", 0, 30, 2)
        diffwalk = st.selectbox("Difficoltà a camminare?", [0, 1])
        education = st.slider("Istruzione (1–6)", 1, 6, 4)
        income = st.slider("Reddito (1–8)", 1, 8, 4)

    b1, b2 = st.columns(2)
    with b1:
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
    with b2:
        altezza_cm = st.number_input("Altezza (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.5)
    altezza_m = max(altezza_cm / 100, 0.0001)
    bmi = peso / (altezza_m ** 2)
    st.markdown(f"**BMI calcolato:** {bmi:.2f}")

    st.write("")
    if st.button("🔎 Calcola risultato", type="primary"):
        record = {
            "HighBP": int(highbp), "HighChol": int(highchol), "CholCheck": int(cholcheck),
            "BMI": round(float(bmi), 1), "Smoker": int(smoker), "Stroke": int(stroke),
            "HeartDiseaseorAttack": int(heartdisease), "PhysActivity": int(physactivity),
            "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(hvyalcoh),
            "AnyHealthcare": int(anyhealthcare), "NoDocbcCost": int(nomedicalcare),
            "GenHlth": int(genhlth), "MentHlth": int(menthlth), "PhysHlth": int(physhlth),
            "DiffWalk": int(diffwalk), "Sex": int(gender), "Age": int(age),
            "Education": int(education), "Income": int(income),
        }
        df = pd.DataFrame([record])

        try:
            model, model_type, meta = load_best_model()
            X = preprocess_for_inference(df, meta)
            pred_class, prob = predict_with_proba(model, model_type, X)
        except Exception as e:
            st.error(f"Errore durante la predizione: {e}")
            return

        st.session_state.last_pred = pred_class
        st.session_state.last_prob = prob

        st.success(f"Risultato: **{pred_class}**  (0=No, 1=Pre, 2=Diabete)")
        st.info(f"Probabilità della classe predetta: **{prob*100:.1f}%**")

        # Prepara messaggio iniziale del bot + vignetta (fallback elegante se LLM non disponibile)
        sys_prompt = build_dynamic_system_prompt(pred_class, prob)
        first_msg = deepseek_chat([], system_prompt=sys_prompt)
        if first_msg.startswith("⚠️") or first_msg.lower().startswith("errore"):
            first_msg = sys_prompt.split("\n")[0]

        st.session_state.messages = [{"role": "assistant", "content": first_msg}]
        st.session_state.chat_open = False
        st.session_state.teaser_message = first_msg
        st.session_state.show_teaser = True

        st.caption("Apri la chat in basso a destra per consigli personalizzati o prenotare una visita.")

# ========== CHAT FLOATING (icona + vignetta + pannello) ==========
def render_chat():
    # — Bottone flottante
    st.markdown("""
    <div id="floating-chat">
      <form action="" method="get">
        <button type="submit" name="chat" value="toggle">💬</button>
      </form>
    </div>
    """, unsafe_allow_html=True)

    # — Toggle / Query params
    qp = st.query_params
    if qp.get("chat") == "toggle":
        if st.session_state.last_pred is None:
            st.session_state.teaser_message = (
                "Per iniziare, compila il form: vedrai il risultato e potrai parlare con l’assistente."
            )
            st.session_state.show_teaser = True
            st.session_state.chat_open = False
        else:
            st.session_state.chat_open = not st.session_state.chat_open
            if st.session_state.chat_open:
                st.session_state.show_teaser = False
        st.query_params.clear()

    # — Vignetta: prima del form → pulsante verso il form; dopo la predizione → 'Apri chat'
    if st.session_state.show_teaser and not st.session_state.chat_open and st.session_state.teaser_message:
        teaser_text = st.session_state.teaser_message
        if len(teaser_text) > 240:
            teaser_text = teaser_text[:235].rstrip() + "…"

        if st.session_state.last_pred is None:
            st.markdown(f"""
            <div id="chat-teaser">
              <div>{teaser_text}</div>
              <div class="teaser-actions">
                <form action="" method="get" style="display:inline;">
                  <button type="submit" name="goto" value="form">Compila il form</button>
                </form>
                <form action="" method="get" style="display:inline;">
                  <button type="submit" name="teaser" value="close">Chiudi</button>
                </form>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div id="chat-teaser">
              <div>{teaser_text}</div>
              <div class="teaser-actions">
                <form action="" method="get" style="display:inline;">
                  <button type="submit" name="chat" value="open">Apri chat</button>
                </form>
                <form action="" method="get" style="display:inline;">
                  <button type="submit" name="teaser" value="close">Chiudi</button>
                </form>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # — Gestione bottoni della vignetta
    if qp.get("chat") == "open":
        if st.session_state.last_pred is None:
            st.session_state.teaser_message = (
                "Prima compila il form per ottenere il risultato. Poi potrai parlare con l’assistente."
            )
            st.session_state.show_teaser = True
            st.session_state.chat_open = False
        else:
            st.session_state.chat_open = True
            st.session_state.show_teaser = False
        st.query_params.clear()

    if qp.get("goto") == "form":
        go("form")
        st.session_state.show_teaser = False
        st.query_params.clear()

    if qp.get("teaser") == "close":
        st.session_state.show_teaser = False
        st.query_params.clear()

    # — Pannello chat (solo dopo una predizione)
    if st.session_state.chat_open and st.session_state.last_pred is not None:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        st.markdown("**Assistente**")
        st.caption("Questo assistente non sostituisce un consulto medico.")

        # cronologia
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # input
        if prompt := st.chat_input("Scrivi un messaggio…"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 1) se sembra un comune → contatti
            reply = None
            comune_candidate = prompt.strip()
            if len(comune_candidate) >= 2 and any(c.isalpha() for c in comune_candidate):
                contacts = get_nearby_contacts(comune_candidate)
                if contacts:
                    lines = ["**Contatti utili nella tua zona**:"]
                    for c in contacts:
                        lines.append(f"- **{c['struttura']}** — {c['telefono']}  \n  _{c['tipo']}_")
                    reply = "\n".join(lines)

            # 2) fallback all'LLM (con messaggio di servizio se indisponibile)
            if not reply:
                sys_prompt = build_dynamic_system_prompt(st.session_state.last_pred or 0, st.session_state.last_prob or 0.0)
                llm_reply = deepseek_chat(st.session_state.messages, system_prompt=sys_prompt)
                if llm_reply.startswith("⚠️") or llm_reply.lower().startswith("errore"):
                    llm_reply = (
                        "Al momento il consulente virtuale non è disponibile. "
                        "Posso comunque aiutarti a cercare i contatti: scrivi il tuo **comune**."
                    )
                reply = llm_reply

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

        st.markdown('</div>', unsafe_allow_html=True)

# ========== ROUTING ==========
if st.session_state.view == "home":
    render_home()
elif st.session_state.view == "form":
    render_form()
else:
    go("home"); render_home()

# chat sempre visibile
render_chat()
