# streamlit_prod/main_prod.py
import sys
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ========== PATH & IMPORT ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_best_model, preprocess_for_inference  # noqa: E402

# ========== PAGE CONFIG & THEME ==========
st.set_page_config(page_title="Valutazione Rischio Diabete", page_icon="🩺", layout="wide")
st.markdown("""
<style>
/* Layout più largo e con meno margini laterali */
html, body { background: #f6f7fb; }
div.block-container{
  padding: 1.25rem 1rem !important;
  max-width: 1400px !important;
}

/* Nascondi sidebar/header icona */
section[data-testid="stSidebar"]{display:none}
header [data-testid="baseButton-headerNoPadding"]{visibility:hidden}

/* Hero */
.hero{
  border-radius: 28px;
  padding: clamp(1.2rem, 2vw, 2rem);
  border: 1px solid rgba(16,24,40,.08);
  background:
    radial-gradient(1200px 600px at 8% 10%, rgba(0,120,255,.08), transparent 60%),
    radial-gradient(1000px 500px at 90% 30%, rgba(255,60,140,.08), transparent 60%),
    linear-gradient(180deg, #ffffff, #fafbff);
  box-shadow: 0 16px 42px rgba(16,24,40,.08);
  margin-bottom: 1rem;
}
.hero h1{ margin: 0 0 .35rem }
.hero p { margin: 0; opacity: .9 }

/* Card pulite */
.card{
  border: 1px solid rgba(16,24,40,.08);
  border-radius: 20px;
  background: #fff;
  box-shadow: 0 8px 22px rgba(16,24,40,.06);
  padding: 1.1rem 1.25rem;
  transition: transform .12s ease, box-shadow .12s ease;
  margin-bottom: 1rem;
}
.card:hover{ transform: translateY(-1px); box-shadow: 0 12px 28px rgba(16,24,40,.08); }

/* Badge probabilità — testo nero come richiesto */
.badge{
  display:inline-block; padding:.34rem .7rem; border-radius:999px;
  border:1px solid rgba(16,24,40,.12); background:#fff; font-size:.8rem;
  color:#111; font-weight:700;
}

/* Pulsanti */
.stButton>button[kind="primary"]{
  border-radius: 14px; font-weight: 700; padding: .78rem 1.05rem;
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
.contact-grid{ display:grid; grid-template-columns:1fr; gap:1rem; }
@media (min-width: 780px){ .contact-grid{ grid-template-columns: 1fr 1fr; } }

/* Lista comuni (pill) */
.pill{
  display:inline-block; margin:.25rem; padding:.35rem .7rem; border-radius:999px;
  border:1px solid rgba(16,24,40,.15); background:#fff; cursor:pointer; font-size:.92rem;
}
.pill:hover{ background:#f2f4f7 }

/* Dark mode */
@media (prefers-color-scheme: dark){
  html, body { background: #0f1117; }
  .hero{ border-color: rgba(255,255,255,.08); background: linear-gradient(180deg,#0f1117,#12131a); color:#e9e9ea }
  .card{ background:#12131a; color:#e9e9ea; border-color: rgba(255,255,255,.08) }
  .prob-wrap{ border-color: rgba(255,255,255,.1); background: #141722; }
}
</style>
""", unsafe_allow_html=True)

# ========== CONTATTI: LETTURA CSV E NORMALIZZAZIONE ==========
CONTACTS_CSV = PROJECT_ROOT / "data" / "ospedali_milano_comuni_mapping.csv"
MILANO_LAT, MILANO_LON = 45.4642, 9.1900  # centro mappa

def _slug(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def _norm_text(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "comune": "comune",
        "citta": "comune",
        "comuni": "comune",
        "ospedale_di_riferimento_linea_daria_macro_area": "ospedale",
        "ospedale_di_riferimento_linea_d_aria_macro_area": "ospedale",
        "ospedale_di_riferimento": "ospedale",
        "ospedale_di_riferimento_macro_area": "ospedale",
        "ospedale_di_riferimento_linea_daria": "ospedale",
        "indirizzo_ospedale": "indirizzo",
        "indirizzo": "indirizzo",
        "telefono": "telefono",
        "prenotazioni_cup": "prenotazioni",
        "cup": "prenotazioni",
        "note": "note",
        "lat": "lat", "latitude": "lat", "latitudine": "lat",
        "lon": "lon", "lng": "lon", "longitude": "lon", "longitudine": "lon",
    }
    new_cols = {c: mapping.get(_slug(c), _slug(c)) for c in df.columns}
    df = df.rename(columns=new_cols)
    for c in ["comune","ospedale","indirizzo","telefono","prenotazioni","note"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for req in ["comune","ospedale","indirizzo","telefono"]:
        if req not in df.columns:
            df[req] = pd.NA
    return df

@st.cache_data(show_spinner=False)
def load_contacts():
    if not CONTACTS_CSV.exists():
        return pd.DataFrame(), []
    try:
        df = pd.read_csv(CONTACTS_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(CONTACTS_CSV, encoding="latin-1", dtype=str, keep_default_na=False)
    df = _canonicalize_columns(df)
    comuni = (
        pd.Series(df["comune"].astype(str).str.strip())
        .replace({"": np.nan}).dropna().drop_duplicates()
        .sort_values(key=lambda s: s.str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.casefold())
        .tolist()
    )
    return df, comuni

contacts_df, comuni_options = load_contacts()

# ========== SESSION ==========
st.session_state.setdefault("view", "home")
st.session_state.setdefault("last_pred", None)
st.session_state.setdefault("last_prob", None)
st.session_state.setdefault("selected_comune", None)
def go(view: str): st.session_state.view = view

# ========== MODEL: PREDIZIONE CON PROB ==========
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
    f1, f2, f3 = st.columns([1,1,1.2])
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
        st.button("📍 Vai ai contatti", use_container_width=True, on_click=lambda: go("contacts"))

# ========== CONTATTI ==========
def show_contacts_ui():
    st.markdown("### 📍 Prenota vicino a te", help="Cerca il comune e visualizza l'ospedale di riferimento e i contatti.")
    if contacts_df.empty or not comuni_options:
        st.warning("Contatti non trovati. Verifica il file `data/ospedali_milano_comuni_mapping.csv`.")
        return

    # Ricerca
    q = st.text_input("🔎 Cerca comune", placeholder="Scrivi almeno 2 lettere…")
    if q and len(q.strip()) >= 2:
        qn = _norm_text(q)
        options = [c for c in comuni_options if qn in _norm_text(c)]
    else:
        options = comuni_options

    st.caption(f"Risultati: {len(options)} comuni")
    # Lista risultati come "pill": click → seleziona comune
    if options:
        show_list = options[:60]
        rows = (len(show_list) + 5) // 6
        for r in range(rows):
            cols = st.columns(6, gap="small")
            for i, col in enumerate(cols):
                idx = r*6 + i
                if idx >= len(show_list): break
                label = show_list[idx]
                with col:
                    if st.button(label, key=f"pill-{label}", use_container_width=True):
                        st.session_state.selected_comune = label

    # Dettagli comune selezionato
    if st.session_state.selected_comune:
        sel = st.session_state.selected_comune
        st.markdown(f"#### 📌 Contatti per **{sel}**")
        mask = contacts_df["comune"].astype(str).map(_norm_text).eq(_norm_text(sel))
        sub = contacts_df.loc[mask].copy()
        if sub.empty:
            st.info("Nessun contatto trovato per il comune selezionato.")
            return

        st.markdown('<div class="contact-grid">', unsafe_allow_html=True)
        for _, row in sub.iterrows():
            osp = (row.get("ospedale") or "Ospedale di riferimento").strip()
            tel = str(row.get("telefono", "")).strip()
            ind = str(row.get("indirizzo", "")).strip()
            cup = str(row.get("prenotazioni", "")).strip()
            note = str(row.get("note", "")).strip()
            tel_html = f'<a href="tel:{tel}">{tel}</a>' if tel else ''
            st.markdown(
                f"""
                <div class="card">
                  <div style="font-weight:750; font-size:1.02rem; margin-bottom:.25rem">{osp}</div>
                  <div style="margin:.2rem 0">{'📞 <b>Telefono:</b> ' + tel_html if tel_html else ''}</div>
                  <div style="margin:.2rem 0">{'📍 <b>Indirizzo:</b> ' + ind if ind else ''}</div>
                  <div style="margin:.2rem 0">{'🗓️ <b>Prenotazioni/CUP:</b> ' + cup if cup else ''}</div>
                  <div style="margin:.2rem 0; opacity:.9">{'📝 <b>Note:</b> ' + note if note else ''}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        csv_bytes = sub.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Scarica contatti (CSV)",
            csv_bytes,
            file_name=f"contatti_{_norm_text(sel)}.csv",
            use_container_width=True,
        )

def render_contacts():
    top = st.columns([1,3])[0]
    if top.button("⬅️ Torna alla Home"): go("home"); st.stop()

    st.markdown("## 🗂️ Contatti ospedalieri per comune")

    # Bottone per ricaricare il CSV senza riavviare l'app
    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("🔄 Ricarica contatti"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            global contacts_df, comuni_options
            contacts_df, comuni_options = load_contacts()
            st.success("Contatti ricaricati.")

    # Ricerca + risultati + dettagli
    show_contacts_ui()

    # Mappa (zoom su Milano, puntine rosse; evidenzia selezionato)
    st.markdown("### 🗺️ Mappa (se disponibile)")
    lat_col = next((c for c in contacts_df.columns if c.lower() in ["lat","latitude","latitudine"]), None)
    lon_col = next((c for c in contacts_df.columns if c.lower() in ["lon","lng","longitude","longitudine"]), None)

    def _zoom_for_bbox(df_sel: pd.DataFrame, lat_col: str, lon_col: str) -> float:
        """Heuristica: zoom più vicino quando il comune è selezionato e i punti sono vicini."""
        if df_sel.empty:
            return 11.0
        lat_span = float(df_sel[lat_col].max() - df_sel[lat_col].min())
        lon_span = float(df_sel[lon_col].max() - df_sel[lon_col].min())
        span = max(lat_span, lon_span)
        # Milano ~ 0.1° ~ qualche km; più piccolo → più zoom
        if span < 0.005:   # ~500m
            return 14.5
        if span < 0.01:    # ~1km
            return 14.0
        if span < 0.02:    # ~2km
            return 13.5
        if span < 0.05:    # ~5km
            return 13.0
        if span < 0.1:     # ~10km
            return 12.0
        return 11.0

    if lat_col and lon_col:
        try:
            dfm = contacts_df.copy()
            dfm[lat_col] = pd.to_numeric(dfm[lat_col], errors="coerce")
            dfm[lon_col] = pd.to_numeric(dfm[lon_col], errors="coerce")
            dfm = dfm.dropna(subset=[lat_col, lon_col])
            if not dfm.empty:
                # Layer generale (rosso)
                layer_all = pdk.Layer(
                    "ScatterplotLayer",
                    data=dfm,
                    get_position=[lon_col, lat_col],
                    get_radius=600,
                    get_fill_color=[200, 0, 0, 200],  # rosso
                    pickable=True,
                    radius_min_pixels=4,
                    radius_max_pixels=20,
                )
                layers = [layer_all]

                # Centro e zoom
                if st.session_state.selected_comune:
                    sel = _norm_text(st.session_state.selected_comune)
                    df_sel = dfm[dfm["comune"].astype(str).map(_norm_text).eq(sel)]
                    if not df_sel.empty:
                        layer_sel = pdk.Layer(
                            "ScatterplotLayer",
                            data=df_sel,
                            get_position=[lon_col, lat_col],
                            get_radius=900,
                            get_fill_color=[255, 120, 0, 230],  # arancio
                            pickable=True,
                            radius_min_pixels=6,
                            radius_max_pixels=24,
                        )
                        layers.append(layer_sel)
                        center_lat = float(df_sel[lat_col].mean())
                        center_lon = float(df_sel[lon_col].mean())
                        zoom = _zoom_for_bbox(df_sel, lat_col, lon_col)  # <--- ZOOM PIÙ VICINO SUL COMUNE
                    else:
                        center_lat, center_lon, zoom = MILANO_LAT, MILANO_LON, 11.0
                else:
                    center_lat, center_lon, zoom = MILANO_LAT, MILANO_LON, 11.0

                view_state = pdk.ViewState(
                    latitude=center_lat, longitude=center_lon,
                    zoom=zoom, pitch=0, bearing=0
                )
                tooltip = {
                    "html": "<b>{comune}</b><br/>{ospedale}<br/>{indirizzo}<br/>{telefono}",
                    "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}
                }
                st.pydeck_chart(
                    pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip),
                    use_container_width=True
                )
            else:
                st.info("Coordinate presenti ma non valide.")
        except Exception as e:
            st.warning(f"Mappa non disponibile: {e}")
    else:
        st.info("Per la mappa, aggiungi colonne **Latitudine/Longitudine** (o lat/lon) nel CSV.")

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
        nomedicalcare=s("Eviti cure per costi?"); genhlth=sl("Salute generale (1–5)",1,5,3)
        menthlth=sl("Giorni problemi mentali (30)",0,30,2); physhlth=sl("Giorni problemi fisici (30)",0,30,2)
        diffwalk=s("Difficoltà a camminare?"); education=sl("Istruzione (1–6)",1,6,4); income=sl("Reddito (1–8)",1,8,4)
    b1,b2=st.columns(2)
    with b1: peso=st.number_input("Peso (kg)",30.0,250.0,70.0,0.5)
    with b2: h=st.number_input("Altezza (cm)",100.0,220.0,170.0,0.5)
    bmi = peso/((max(h,.0001)/100)**2); st.caption(f"👉 BMI: **{bmi:.2f}**")

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

        # Card risultato (colore in base a classe)
        if cls == 2:
            color, label = "#ef4444", "Rischio alto (2)"
        elif cls == 1:
            color, label = "#f59e0b", "Rischio medio (1)"
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

    # Consiglio in base al rischio + tasto contatti
    if st.session_state.last_pred is not None:
        cls = int(st.session_state.last_pred)
        if cls == 0:
            st.info("✅ **Basso rischio** — se vuoi puoi comunque **prenotare una visita di controllo**.")
        else:
            st.warning("⚠️ **Rischio medio/alto** — è **fortemente consigliato** prenotare un **controllo medico**.")

        st.button("🔎 Cerca contatti ospedale", use_container_width=True, on_click=lambda: go("contacts"))

# ========== ROUTING ==========
if st.session_state.view == "home":
    render_home()
elif st.session_state.view == "form":
    render_form()
elif st.session_state.view == "contacts":
    render_contacts()
else:
    go("home"); render_home()
