# streamlit_prod/main_prod.py
import sys
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ========== PATH & IMPORT ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_best_model, preprocess_for_inference  # noqa: E402

# ========== PAGE CONFIG & THEME ==========
st.set_page_config(page_title="Valutazione Rischio Diabete", page_icon="🩺", layout="wide")
st.markdown("""
<style>
html, body { background: #f6f7fb; }
div.block-container{ padding: 2.2rem 1rem !important; max-width: 1180px !important; }
section[data-testid="stSidebar"]{display:none}
header [data-testid="baseButton-headerNoPadding"]{visibility:hidden}

/* Hero */
.hero{
  border-radius: 26px;
  padding: clamp(1.4rem, 2.2vw, 2.2rem);
  border: 1px solid rgba(16,24,40,.08);
  background:
    radial-gradient(1200px 600px at 8% 10%, rgba(0,120,255,.08), transparent 60%),
    radial-gradient(1000px 500px at 90% 30%, rgba(255,60,140,.08), transparent 60%),
    linear-gradient(180deg, #ffffff, #fafbff);
  box-shadow: 0 16px 42px rgba(16,24,40,.08);
}
.hero h1{ margin: 0 0 .25rem }
.hero p { margin: 0; opacity: .9 }

/* Card */
.card{
  border: 1px solid rgba(16,24,40,.08);
  border-radius: 18px;
  background: #fff;
  box-shadow: 0 8px 22px rgba(16,24,40,.06);
  padding: 1rem 1.2rem;
  transition: transform .12s ease, box-shadow .12s ease;
}
.card:hover{ transform: translateY(-1px); box-shadow: 0 12px 28px rgba(16,24,40,.08); }
.badge{ display:inline-block; padding:.3rem .65rem; border-radius:999px; border:1px solid rgba(16,24,40,.12); background:#fff; font-size:.78rem }

/* Pulsanti */
.stButton>button[kind="primary"]{
  border-radius: 14px; font-weight: 700; padding: .72rem 1rem;
  box-shadow: 0 12px 26px rgba(16,24,40,.12);
}
.stButton>button{ border-radius: 12px; font-weight: 600 }

/* Barra probabilità */
.prob-wrap{
  width:100%; height: 10px; border-radius: 999px;
  border: 1px solid rgba(16,24,40,.12); background: #f2f4f7; overflow:hidden;
}
.prob-fill{ height:100%; background: linear-gradient(90deg,#60a5fa,#a78bfa); }

/* Risultato card colorata */
.result{ border-left-width: 6px; border-left-style: solid; padding-left: 1rem; }

/* Griglia contatti */
.contact-grid{ display:grid; grid-template-columns:1fr; gap:.9rem; }
@media (min-width: 720px){ .contact-grid{ grid-template-columns: 1fr 1fr; } }

/* Dark */
@media (prefers-color-scheme: dark){
  html, body { background: #0f1117; }
  .hero{ border-color: rgba(255,255,255,.08); background: linear-gradient(180deg,#0f1117,#12131a); color:#e9e9ea }
  .card{ background:#12131a; color:#e9e9ea; border-color: rgba(255,255,255,.08) }
  .prob-wrap{ border-color: rgba(255,255,255,.1); background: #141722; }
}
</style>
""", unsafe_allow_html=True)

# ========== NORMALIZZAZIONE COLONNE CONTATTI ==========
CONTACTS_CSV = PROJECT_ROOT / "data" / "ospedali_milano_comuni_mapping.csv"

def _slug(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")  # rimuove accenti/es. d’aria -> daria
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Mappa robusta dai nomi "slug" del CSV alla nostra convenzione
    mapping = {
        "comune": "comune",
        "citta": "comune",
        "comuni": "comune",
        "ospedale_di_riferimento_linea_daria_macro_area": "ospedale",
        "ospedale_di_riferimento_linea_d_aria_macro_area": "ospedale",
        "ospedale_di_riferimento": "ospedale",
        "struttura": "ospedale",
        "indirizzo_ospedale": "indirizzo",
        "indirizzo": "indirizzo",
        "telefono": "telefono",
        "prenotazioni_cup": "prenotazioni",
        "cup": "prenotazioni",
        "note": "note",
        "tipo": "note",
    }
    new_cols = {}
    for c in df.columns:
        slug = _slug(c)
        new_cols[c] = mapping.get(slug, slug)  # se non mappato, tiene lo slug
    df = df.rename(columns=new_cols)
    # Garantisco colonne chiave se non presenti
    for req in ["comune", "ospedale", "indirizzo", "telefono"]:
        if req not in df.columns:
            df[req] = pd.NA
    return df

@st.cache_data(show_spinner=False)
def load_contacts():
    if not CONTACTS_CSV.exists():
        return pd.DataFrame(), []
    # utf-8-sig per eventuale BOM da Excel; dtype=str per non perdere zeri nei telefoni
    df = pd.read_csv(CONTACTS_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = _canonicalize_columns(df)
    # pulizia base
    for c in ["comune","ospedale","indirizzo","telefono","prenotazioni","note"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    comuni = sorted(x for x in df["comune"].unique() if str(x).strip())
    return df, comuni

contacts_df, comuni_options = load_contacts()

# ========== SESSION ==========
st.session_state.setdefault("view", "home")
st.session_state.setdefault("last_pred", None)
st.session_state.setdefault("last_prob", None)
st.session_state.setdefault("selected_comune", None)
def go(view: str): st.session_state.view = view

# ========== MODEL UTILS ==========
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
    # Keras
    p = model.predict(X, verbose=0)[0]; c = int(np.argmax(p)); return c, float(p[c])

# ========== HOME ==========
def render_home():
    st.markdown(
        """<div class="hero">
           <h1>🩺 Valutazione del Rischio Diabete</h1>
           <p>Compila il form per ottenere la <b>classe (0/1/2)</b> e la <b>probabilità</b>. Poi seleziona il tuo <b>comune</b> per contatti utili.</p>
           </div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    f1, f2, f3 = st.columns([1,1,1])
    with f1:
        st.markdown("#### 🎯 Perché")
        st.markdown("- Screening rapido\n- Probabilità oltre la classe\n- Supporto pratico")
    with f2:
        st.markdown("#### 🔒 Privacy")
        st.markdown("I dati sono usati solo per questa valutazione.\nNon sostituisce un consulto medico.")
    with f3:
        st.markdown("#### ⚙️ Come funziona")
        st.markdown("1) Compila il form\n2) Vedi risultato + probabilità\n3) Seleziona comune per contatti")
    st.write("")
    cta1, cta2, _ = st.columns([1,1,1])
    with cta1:
        st.button("📝 Apri il form", type="primary", use_container_width=True, on_click=lambda: go("form"))
    with cta2:
        st.button("📍 Vai ai contatti", use_container_width=True, on_click=lambda: go("form"))

# ========== FORM ==========
def render_form():
    top = st.columns([1,3])[0]
    if top.button("⬅️ Torna alla Home"): go("home"); st.stop()

    st.markdown("### 📋 Valutazione personale")
    st.caption("Compila i campi. Il BMI è calcolato automaticamente.")

    s=lambda t: st.selectbox(t,[0,1]); sl=lambda t,a,b,d: st.slider(t,a,b,d)
    c1,c2=st.columns(2)
    with c1:
        gender=s("Sesso (0=femmina, 1=maschio)"); age=sl("Età",18,90,40)
        highbp=s("Pressione alta?"); highchol=s("Colesterolo alto?"); cholcheck=s("Controllo colesterolo (5 anni)?")
        smoker=s("Fumi?"); stroke=s("Ictus in passato?"); heartdisease=s("Malattie cardiache?")
        physactivity=s("Attività fisica regolare?"); fruits=s("Frutta regolare?")
    with c2:
        veggies=s("Verdura regolare?"); hvyalcoh=s("Consumo elevato di alcol?"); anyhealthcare=s("Accesso a servizi sanitari?")
        nomedicalcare=s("Eviti cure per costi?"); genhlth=sl("Salute generale (1 ottima – 5 pessima)",1,5,3)
        menthlth=sl("Giorni problemi mentali (30)",0,30,2); physhlth=sl("Giorni problemi fisici (30)",0,30,2)
        diffwalk=s("Difficoltà a camminare?"); education=sl("Istruzione (1–6)",1,6,4); income=sl("Reddito (1–8)",1,8,4)
    b1,b2=st.columns(2)
    with b1: peso=st.number_input("Peso (kg)",30.0,250.0,70.0,0.5)
    with b2: h=st.number_input("Altezza (cm)",100.0,220.0,170.0,0.5)
    bmi = peso/((max(h,.0001)/100)**2); st.caption(f"👉 BMI: **{bmi:.2f}**")

    # Calcolo e risultato
    if st.button("🔎 Calcola risultato", type="primary", use_container_width=True):
        rec = pd.DataFrame([{
            "HighBP":int(highbp),"HighChol":int(highchol),"CholCheck":int(cholcheck),"BMI":round(float(bmi),1),
            "Smoker":int(smoker),"Stroke":int(stroke),"HeartDiseaseorAttack":int(heartdisease),"PhysActivity":int(physactivity),
            "Fruits":int(fruits),"Veggies":int(veggies),"HvyAlcoholConsump":int(hvyalcoh),"AnyHealthcare":int(anyhealthcare),
            "NoDocbcCost":int(nomedicalcare),"GenHlth":int(genhlth),"MentHlth":int(menthlth),"PhysHlth":int(physhlth),
            "DiffWalk":int(diffwalk),"Sex":int(gender),"Age":int(age),"Education":int(education),"Income":int(income),
        }])

        try:
            model, model_type, meta = load_best_model()
            X = preprocess_for_inference(rec, meta)
            cls, prob = predict_with_proba(model, model_type, X)
        except Exception as e:
            st.error(f"Errore durante la predizione: {e}")
            return

        st.session_state.last_pred = cls
        st.session_state.last_prob = prob

        # Card risultato (colore in base a classe/prob)
        if cls == 2:
            color, label = "#ef4444", "Rischio alto (2)"
        elif cls == 1 and prob >= 0.65:
            color, label = "#f59e0b", "Pre-diabete alto (1)"
        elif cls == 1:
            color, label = "#fbbf24", "Pre-diabete (1)"
        else:
            color, label = "#10b981", "Basso rischio (0)"

        prob_pct = int(prob * 100)
        st.markdown(
            f"""
            <div class="card result" style="border-left-color:{color}">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
                <div style="font-weight:800;font-size:1.05rem">Risultato: {label}</div>
                <span class="badge">Probabilità: {prob_pct}%</span>
              </div>
              <div style="margin-top:.6rem" class="prob-wrap">
                <div class="prob-fill" style="width:{prob_pct}%"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.toast("Consiglio disponibile più sotto in base al risultato.", icon="💡")

    # — Consigli sintetici (inline, sempre visibili dopo un risultato)
    if st.session_state.last_pred is not None:
        cls = st.session_state.last_pred
        prob = float(st.session_state.last_prob or 0)
        with st.container():
            if cls == 2:
                st.info("🔔 **Consiglio**: rischio compatibile con diabete → contatta il **medico di base** o una **struttura ospedaliera** per una visita.")
            elif cls == 1 and prob >= 0.65:
                st.warning("⚠️ **Consiglio**: condizione pre-diabetica **probabilità elevata** → prenota un controllo e cura stile di vita.")
            elif cls == 1:
                st.info("ℹ️ **Consiglio**: condizione pre-diabetica → stile di vita sano e **controllo non urgente**.")
            else:
                st.success("✅ **Consiglio**: basso rischio → mantieni uno stile equilibrato e fai controlli periodici.")

    # — Sezione contatti (dopo calcolo)
    st.write("")
    st.markdown("### 📍 Prenota vicino a te")
    if contacts_df.empty or not comuni_options:
        st.warning("Contatti non trovati. Verifica il file `data/ospedali_milano_comuni_mapping.csv`.")
        return

    disable_select = st.session_state.last_pred is None
    placeholder = "Cerca il tuo comune…" if not disable_select else "Prima calcola il risultato"
    selected = st.selectbox("Seleziona il comune", comuni_options, index=None,
                            placeholder=placeholder, disabled=disable_select)
    if selected:
        st.session_state.selected_comune = selected

    if st.session_state.selected_comune:
        sub = contacts_df[contacts_df["comune"].astype(str).str.lower()
                          == st.session_state.selected_comune.lower()]
        if sub.empty:
            st.info("Nessun contatto trovato per il comune selezionato.")
            return

        st.markdown("#### Strutture e contatti")
        st.markdown('<div class="contact-grid">', unsafe_allow_html=True)
        for _, row in sub.iterrows():
            osp = str(row.get("ospedale", "")).strip()
            ind = str(row.get("indirizzo", "")).strip()
            tel = str(row.get("telefono", "")).strip()
            cup = str(row.get("prenotazioni", "")).strip()
            note = str(row.get("note", "")).strip()

            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;gap:8px;align-items:center">
                    <div style="font-weight:750">{osp or 'Ospedale di riferimento'}</div>
                    {'<span class="badge">CUP</span>' if cup else ''}
                  </div>
                  <div style="opacity:.9;margin-top:.35rem">{ind or '—'}</div>
                  <div style="margin-top:.35rem">{'📞 '+tel if tel else ''}</div>
                  <div style="margin-top:.35rem">{'🗓️ '+cup if cup else ''}</div>
                  <div style="margin-top:.35rem;opacity:.85">{'📝 '+note if note else ''}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        csv_bytes = sub.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Scarica contatti (CSV)", csv_bytes,
                           file_name=f"contatti_{st.session_state.selected_comune}.csv", use_container_width=True)

# ========== ROUTING ==========
if st.session_state.view == "home":
    render_home()
elif st.session_state.view == "form":
    render_form()
else:
    go("home"); render_home()
