# streamlit_prod/main_prod.py
import sys
from pathlib import Path
from datetime import datetime, timezone

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
section[data-testid="stSidebar"]{display:none}
header [data-testid="baseButton-headerNoPadding"]{visibility:hidden}
div.block-container{padding:2rem 0}
.hero{
  border-radius:28px;padding:clamp(1.6rem,2.5vw,2.6rem);
  border:1px solid rgba(0,0,0,.06);
  background:
    radial-gradient(1200px 600px at 8% 10%, rgba(0,120,255,.10), transparent 60%),
    radial-gradient(1000px 500px at 90% 30%, rgba(255,60,140,.10), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,.94), rgba(255,255,255,.88));
  box-shadow:0 14px 44px rgba(0,0,0,.08);
}
.hero h1{margin:0 0 .5rem}
.card{
  border:1px solid rgba(0,0,0,.08); background:#fff;
  border-radius:18px; box-shadow:0 8px 22px rgba(0,0,0,.06);
  padding:1rem 1.2rem; margin-bottom:.8rem;
}
.badge{display:inline-block;padding:.25rem .6rem;border-radius:999px;border:1px solid rgba(0,0,0,.1);font-size:.8rem;background:#fff}
.stButton>button[kind="primary"]{border-radius:14px;font-weight:700;padding:.7rem 1rem;box-shadow:0 10px 24px rgba(0,0,0,.10)}
.stButton>button{border-radius:12px;font-weight:600}
@media (prefers-color-scheme: dark){
  .hero{border-color:rgba(255,255,255,.08);background:linear-gradient(180deg,#0f1117,#0f121a);color:#e9e9ea}
  .card{background:#101318;color:#e9e9ea;border-color:rgba(255,255,255,.08)}
}
</style>
""", unsafe_allow_html=True)

# ========== COSTANTI/CONTATTI ==========
CONTACTS_CSV = PROJECT_ROOT / "data" / "ospedali_milano_comuni_mapping.csv"

@st.cache_data(show_spinner=False)
def load_contacts():
    if not CONTACTS_CSV.exists():
        return pd.DataFrame(), []
    df = pd.read_csv(CONTACTS_CSV)
    df.columns = [c.lower() for c in df.columns]
    rename_map = {"città":"comune", "citta":"comune", "comuni":"comune"}
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    comuni = sorted(df["comune"].dropna().astype(str).unique()) if "comune" in df.columns else []
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
    p = model.predict(X, verbose=0)[0]; c = int(np.argmax(p)); return c, float(p[c])

# ========== POPUP RACCOMANDAZIONI ==========
def show_recommendation_popup(cls: int, prob: float):
    title = "Consiglio medico"
    msg = ""
    if cls == 2:
        msg = ("La valutazione indica **rischio compatibile con diabete**.\n\n"
               "👉 Ti consigliamo di **contattare il tuo medico di base** o una **struttura ospedaliera** per una visita approfondita.")
    elif cls == 1 and prob >= 0.65:
        msg = ("La valutazione indica **condizione pre-diabetica** con probabilità **elevata**.\n\n"
               "👉 È opportuno **prenotare un controllo** e monitorare dieta e attività fisica.")
    elif cls == 1:
        msg = ("La valutazione indica **condizione pre-diabetica**.\n\n"
               "👉 Consigliamo **stile di vita sano** e un **controllo medico** non urgente.")
    else:
        msg = ("La valutazione indica **basso rischio**.\n\n"
               "👉 Mantieni uno stile di vita equilibrato e considera **controlli periodici**.")

    # Modal (Streamlit 1.32+). Fallback su info se non disponibile.
    try:
        dlg = getattr(st, "experimental_dialog", None) or getattr(st, "dialog", None)
        if dlg:
            @dlg(title)
            def _popup():
                st.write(msg)
                if st.session_state.selected_comune and not contacts_df.empty:
                    sub = contacts_df[contacts_df["comune"].astype(str).str.lower() ==
                                      st.session_state.selected_comune.lower()]
                    if not sub.empty:
                        st.write("**Contatti vicini a te:**")
                        st.dataframe(sub[["struttura","indirizzo","telefono"]], use_container_width=True)
                st.button("Chiudi")
            _popup()
        else:
            st.info(msg)
    except Exception:
        st.info(msg)

# ========== HOME ==========
def render_home():
    st.markdown(
        """<div class="hero">
           <h1>🩺 Valutazione del Rischio Diabete</h1>
           <p class="small">Compila il form: ottieni la <b>classe (0/1/2)</b> e la <b>probabilità</b>. Poi scegli il tuo <b>comune</b> per contatti utili.</p>
           </div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2, col3 = st.columns([1,1,1.2])
    with col1:
        st.markdown("### 🎯 Perché\n- Screening rapido\n- Probabilità oltre la classe\n- Supporto pratico")
    with col2:
        st.markdown("### 🔒 Privacy\nI dati sono usati solo per questa valutazione.\nNon sostituisce un consulto medico.")
    with col3:
        st.markdown("### ⚙️ Come funziona\n1) Compila il form\n2) Vedi risultato+prob\n3) Seleziona comune per contatti")
    st.write("")
    if st.button("📝 Apri il form", type="primary", use_container_width=True):
        go("form")

# ========== FORM ==========
def render_form():
    top = st.columns([1,3])[0]
    if top.button("⬅️ Torna alla Home"): go("home"); st.stop()

    st.title("📋 Valutazione personale")
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

    if st.button("🔎 Calcola risultato", type="primary"):
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

        st.success(f"Risultato: **{cls}**  (0=No, 1=Pre, 2=Diabete)")
        st.info(f"Probabilità della classe predetta: **{prob*100:.1f}%**")

        # Mostra popup raccomandazioni contestuali
        show_recommendation_popup(cls, prob)

    # — Sezione contatti (post-calcolo)
    st.write("---")
    st.subheader("📍 Prenota vicino a te")
    if contacts_df.empty or not comuni_options:
        st.warning("Database contatti non trovato. Assicurati di avere `data/ospedali_milano_comuni_mapping.csv`.")
        return

    disable_select = st.session_state.last_pred is None
    placeholder = "Cerca il tuo comune…" if not disable_select else "Prima calcola il risultato"
    selected = st.selectbox("Seleziona il comune", comuni_options, index=None, placeholder=placeholder, disabled=disable_select)
    if selected:
        st.session_state.selected_comune = selected

    if st.session_state.selected_comune:
        sub = contacts_df[contacts_df["comune"].astype(str).str.lower() == st.session_state.selected_comune.lower()]
        if sub.empty:
            st.info("Nessun contatto trovato per il comune selezionato.")
            return

        st.markdown("#### Strutture e contatti")
        for _, row in sub.iterrows():
            struttura = str(row.get("struttura", "Struttura")).strip()
            indirizzo = str(row.get("indirizzo", "")).strip()
            telefono  = str(row.get("telefono", "")).strip()
            tipo      = str(row.get("tipo", "")).strip()

            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;gap:8px;align-items:center">
                    <div style="font-weight:700">{struttura}</div>
                    {'<span class="badge">'+tipo+'</span>' if tipo else ''}
                  </div>
                  <div style="opacity:.9;margin-top:.25rem">{indirizzo or '—'}</div>
                  <div style="margin-top:.25rem">{'📞 '+telefono if telefono else ''}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        csv_bytes = sub.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Scarica contatti (CSV)", csv_bytes, file_name=f"contatti_{st.session_state.selected_comune}.csv")

# ========== ROUTING ==========
if st.session_state.view == "home":
    render_home()
elif st.session_state.view == "form":
    render_form()
else:
    go("home"); render_home()
