# streamlit_prod/main_streamlit_prod.py
import sys
import re
import unicodedata
import csv
import os
import base64
import json
import requests
from uuid import uuid4
from datetime import datetime
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
html, body { background: #f6f7fb; }
div.block-container{ padding: 1.1rem .8rem !important; max-width: 1480px !important; }
section[data-testid="stSidebar"]{display:none}
header [data-testid="baseButton-headerNoPadding"]{visibility:hidden}

/* ------- WOW HOME ------- */
.hero-wow{
  position: relative; border-radius: 28px; padding: clamp(1.4rem, 2.2vw, 2.2rem);
  overflow: hidden; border: 1px solid rgba(16,24,40,.08);
  background:
    radial-gradient(1200px 600px at 8% 10%, rgba(56,189,248,.18), transparent 60%),
    radial-gradient(900px 500px at 92% 30%, rgba(168,85,247,.18), transparent 60%),
    linear-gradient(180deg, #ffffff, #f9fbff);
  box-shadow: 0 18px 46px rgba(16,24,40,.10); margin-bottom: 1.1rem;
}
.hero-wow:before{ content: ""; position:absolute; inset:-20%;
  background: conic-gradient(from 180deg at 50% 50%, rgba(99,102,241,.1), rgba(56,189,248,.08), rgba(236,72,153,.08), rgba(99,102,241,.1));
  filter: blur(40px); opacity:.6; }
.hero-inner{ position: relative; z-index: 2; }
.gradient-title{
  font-weight: 900; letter-spacing: .2px; line-height: 1.08;
  font-size: clamp(1.8rem, 3.4vw, 3.2rem);
  background: linear-gradient(90deg, #0ea5e9, #6366f1, #f43f5e, #0ea5e9);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  background-size: 300% 100%; animation: shimmer 8s linear infinite; margin: 0 0 .4rem;
}
@keyframes shimmer { 0% { background-position: 0% 50% } 100% { background-position: 300% 50% } }
.subtitle{ margin: 0; opacity: .9; font-size: clamp(.98rem, 1.4vw, 1.05rem); }

/* Feature cards */
.feature-grid{ display:grid; grid-template-columns:1fr; gap:.9rem; }
@media (min-width: 880px){ .feature-grid{ grid-template-columns: 1fr 1fr 1fr; } }
.card{
  border: 1px solid rgba(16,24,40,.08); border-radius: 20px; background: #fff;
  box-shadow: 0 8px 22px rgba(16,24,40,.06); padding: 1.05rem 1.2rem;
  transition: transform .14s ease, box-shadow .14s ease;
}
.card:hover{ transform: translateY(-1px); box-shadow: 0 14px 28px rgba(16,24,40,.08); }
.card .icon{ font-size: 1.4rem; }

.badge{
  display:inline-block; padding:.34rem .7rem; border-radius:999px;
  border:1px solid rgba(16,24,40,.12); background:#fff; font-size:.8rem;
  color:#111; font-weight:700;
}

/* Pulsanti */
.stButton>button[kind="primary"]{
  border-radius: 14px; font-weight: 800; padding: .82rem 1.1rem;
  box-shadow: 0 14px 30px rgba(16,24,40,.14);
}
.stButton>button{ border-radius: 12px; font-weight: 650 }

/* Barra probabilità */
.prob-wrap{ width:100%; height: 10px; border-radius: 999px; border: 1px solid rgba(16,24,40,.12); background: #f2f4f7; overflow:hidden; }
.prob-fill{ height:100%; background: linear-gradient(90deg,#60a5fa,#a78bfa); }

/* Risultato card colorata */
.result{ border-left-width: 6px; border-left-style: solid; padding-left: 1rem; }

/* Griglia contatti */
.contact-grid{ display:grid; grid-template-columns:1fr; gap:1rem; }
@media (min-width: 780px){ .contact-grid{ grid-template-columns: 1fr 1fr; } }

/* Row azioni centrata */
.action-center{ display:flex; justify-content:center; gap:1rem; margin:.6rem 0 0; }

@media (prefers-color-scheme: dark){
  html, body { background: #0f1117; }
  .hero-wow{ border-color: rgba(255,255,255,.08); background: linear-gradient(180deg,#0f1117,#12131a); color:#e9e9ea }
  .card{ background:#12131a; color:#e9e9ea; border-color: rgba(255,255,255,.08) }
  .prob-wrap{ border-color: rgba(255,255,255,.1); background: #141722; }
}
</style>
""", unsafe_allow_html=True)

# ========== FILES & CONFIG ==========
CONTACTS_CSV = PROJECT_ROOT / "data" / "ospedali_milano_comuni_mapping.csv"
LOG_CSV      = PROJECT_ROOT / "data" / "prod_interactions.csv"  # log interazioni
MILANO_LAT, MILANO_LON = 45.4642, 9.1900

# --- GitHub auto-push config (opzionale) ---
GH_PUSH_ENABLED = os.getenv("GH_PUSH_ENABLED", "0").lower() in {"1", "true", "yes"}
GH_TOKEN   = os.getenv("GITHUB_TOKEN", "")
GH_REPO    = os.getenv("GITHUB_REPO", "KirollosSeif/Diabete")
GH_BRANCH  = os.getenv("GITHUB_BRANCH", "main")
GH_LOG_PATH = os.getenv("GITHUB_LOG_PATH", "data/prod_interactions.csv")

# ========== UTILS NORMALIZZAZIONE ==========
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
    # Mappa colonne del CSV fornito:
    # Comune, "Ospedale di riferimento (linea d’aria, macro-area)", Indirizzo Ospedale, Telefono, Prenotazioni/CUP, Note
    mapping = {
        "comune": "comune", "citta": "comune", "comuni": "comune",
        "ospedale_di_riferimento_linea_daria_macro_area": "ospedale",
        "ospedale_di_riferimento_macro_area": "ospedale",
        "ospedale_di_riferimento": "ospedale",
        "ospedale": "ospedale",
        "indirizzo_ospedale": "indirizzo", "indirizzo": "indirizzo",
        "telefono": "telefono",
        "prenotazioni_cup": "prenotazioni", "prenotazioni": "prenotazioni", "cup": "prenotazioni",
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
st.session_state.setdefault("last_form", None)
st.session_state.setdefault("selected_comune", None)
st.session_state.setdefault("sid", str(uuid4()))

def go(view: str): st.session_state.view = view

# ========== LOG CSV ==========
LOG_COLUMNS = [
    "timestamp","session_id","event_type",
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth",
    "DiffWalk","Sex","Age","Education","Income",
    "predicted_class","probability",
    "comune","ospedale","telefono","indirizzo","prenotazioni","note"
]

def _gh_headers():
    return {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def _gh_contents_url(path_in_repo: str) -> str:
    return f"https://api.github.com/repos/{GH_REPO}/contents/{path_in_repo}"

def push_file_to_github(local_path: Path, path_in_repo: str = GH_LOG_PATH, message: str = "Update prod_interactions.csv") -> bool:
    """Carica/aggiorna il file su GitHub (branch GH_BRANCH). Ritorna True se ok, False altrimenti."""
    if not GH_PUSH_ENABLED or not GH_TOKEN:
        return False
    if not local_path.exists():
        return False
    try:
        content_bytes = local_path.read_bytes()
        content_b64 = base64.b64encode(content_bytes).decode("utf-8")

        sha = None
        r_get = requests.get(_gh_contents_url(path_in_repo), params={"ref": GH_BRANCH}, headers=_gh_headers(), timeout=15)
        if r_get.status_code == 200:
            sha = r_get.json().get("sha")

        payload = {"message": message, "content": content_b64, "branch": GH_BRANCH}
        if sha:
            payload["sha"] = sha

        r_put = requests.put(_gh_contents_url(path_in_repo), headers=_gh_headers(), data=json.dumps(payload), timeout=20)
        return 200 <= r_put.status_code < 300
    except Exception:
        return False

def append_log_row(row: dict):
    """Accoda una riga a data/prod_interactions.csv (crea file se non esiste) + tenta push su GitHub."""
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_CSV.exists() or LOG_CSV.stat().st_size == 0
    safe = {k: row.get(k, "") for k in LOG_COLUMNS}
    with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(safe)
        # tenta push (non blocca UI)
        try:
            _ = push_file_to_github(LOG_CSV)
        except Exception:
            pass

def build_form_dict(**vals) -> dict:
    """Ritorna i campi del form con BMI calcolato."""
    bmi = vals["peso"] / max((vals["altezza_cm"]/100.0)**2, 1e-6)
    return {
        "HighBP":int(vals["highbp"]), "HighChol":int(vals["highchol"]), "CholCheck":int(vals["cholcheck"]),
        "BMI":round(float(bmi),1), "Smoker":int(vals["smoker"]), "Stroke":int(vals["stroke"]),
        "HeartDiseaseorAttack":int(vals["heartdisease"]), "PhysActivity":int(vals["physactivity"]),
        "Fruits":int(vals["fruits"]), "Veggies":int(vals["veggies"]), "HvyAlcoholConsump":int(vals["hvyalcoh"]),
        "AnyHealthcare":int(vals["anyhealthcare"]), "NoDocbcCost":int(vals["nomedicalcare"]),
        "GenHlth":int(vals["genhlth"]), "MentHlth":int(vals["menthlth"]), "PhysHlth":int(vals["physhlth"]),
        "DiffWalk":int(vals["diffwalk"]), "Sex":int(vals["gender"]), "Age":int(vals["age"]),
        "Education":int(vals["education"]), "Income":int(vals["income"]),
    }

def log_prediction(form_d: dict, cls: int, prob: float):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": st.session_state.sid,
        "event_type": "prediction",
        **form_d,
        "predicted_class": int(cls),
        "probability": round(float(prob), 4),
        "comune": "", "ospedale": "", "telefono": "", "indirizzo": "", "prenotazioni": "", "note": ""
    }
    append_log_row(row)

def log_contact_view(comune: str, contact_row: dict):
    form_d = st.session_state.last_form or {}
    cls = st.session_state.last_pred
    prob = st.session_state.last_prob
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": st.session_state.sid,
        "event_type": "contact_view",
        **{k: form_d.get(k, "") for k in [
            "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
            "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth",
            "DiffWalk","Sex","Age","Education","Income"
        ]},
        "predicted_class": (int(cls) if cls is not None else ""),
        "probability": (round(float(prob), 4) if prob is not None else ""),
        "comune": comune,
        "ospedale": str(contact_row.get("ospedale","")).strip(),
        "telefono": str(contact_row.get("telefono","")).strip(),
        "indirizzo": str(contact_row.get("indirizzo","")).strip(),
        "prenotazioni": str(contact_row.get("prenotazioni","")).strip(),
        "note": str(contact_row.get("note","")).strip(),
    }
    append_log_row(row)

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
        """<div class="hero-wow">
            <div class="hero-inner">
              <div class="gradient-title">Valutazione del Rischio Diabete</div>
              <p class="subtitle">Predizione <b>0/1/2</b> con probabilità. Contatti rapidi agli ospedali del tuo comune.</p>
            </div>
           </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    st.markdown(
        """<div class="card"><div class="icon">⚡</div>
           <div style="font-weight:800; font-size:1.05rem; margin-top:.2rem">Valutazione immediata</div>
           <div style="opacity:.9">Inserisci poche informazioni, calcoliamo il <b>BMI</b> e stimiamo il rischio con <b>probabilità</b>.</div>
           </div>""", unsafe_allow_html=True)
    st.markdown(
        """<div class="card"><div class="icon">🧠</div>
           <div style="font-weight:800; font-size:1.05rem; margin-top:.2rem">Modelli ottimizzati</div>
           <div style="opacity:.9">Selezione automatica tra pipeline <b>Sklearn</b> e rete <b>Keras</b> con preprocess dedicato.</div>
           </div>""", unsafe_allow_html=True)
    st.markdown(
        """<div class="card"><div class="icon">🏥</div>
           <div style="font-weight:800; font-size:1.05rem; margin-top:.2rem">Contatti ospedalieri</div>
           <div style="opacity:.9">Cerca il tuo <b>comune</b> e ottieni <b>indirizzo</b>, <b>telefono</b> e <b>Prenotazioni/CUP</b>.</div>
           </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Pulsanti centrati
    st.write("")
    left, center, right = st.columns([1, 1.2, 1], gap="large")
    with center:
        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.button("📝 Apri il form", type="primary", use_container_width=True, on_click=lambda: go("form"))
        with c2:
            st.button("📍 Vai ai contatti", use_container_width=True, on_click=lambda: go("contacts"))

    # (Facoltativo) scarica log
    st.write("---")
    st.caption(f"📄 Log CSV: {LOG_CSV.resolve()}")
    if LOG_CSV.exists():
        st.download_button("⬇️ Scarica log (CSV)",
                           LOG_CSV.read_bytes(),
                           file_name="prod_interactions.csv",
                           mime="text/csv")

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

    # "Pill" di selezione centrata
    if options:
        # Mostra max 60 per riga, in righe multiple
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
        for i, (_, row) in enumerate(sub.iterrows()):
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
            # Logga solo la prima riga (tipicamente un ospedale per comune)
            if i == 0:
                log_contact_view(sel, row.to_dict())

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
        if st.button("🔁 Forza sincronizzazione log su GitHub"):
            ok = push_file_to_github(LOG_CSV)
            st.success("Sincronizzato su GitHub ✅") if ok else st.warning("Sync fallita: controlla token/config.")

    show_contacts_ui()

    # Mappa (zoom su Milano; evidenzia selezionato)
    st.markdown("### 🗺️ Mappa (se disponibile)")
    lat_col = next((c for c in contacts_df.columns if c.lower() in ["lat","latitude","latitudine"]), None)
    lon_col = next((c for c in contacts_df.columns if c.lower() in ["lon","lng","longitude","longitudine"]), None)

    def _zoom_for_bbox(df_sel: pd.DataFrame, lat_col: str, lon_col: str) -> float:
        if df_sel.empty:
            return 11.0
        lat_span = float(df_sel[lat_col].max() - df_sel[lat_col].min())
        lon_span = float(df_sel[lon_col].max() - df_sel[lon_col].min())
        span = max(lat_span, lon_span)
        if span < 0.005: return 14.5
        if span < 0.01:  return 14.0
        if span < 0.02:  return 13.5
        if span < 0.05:  return 13.0
        if span < 0.1:   return 12.0
        return 11.0

    if lat_col and lon_col:
        try:
            dfm = contacts_df.copy()
            dfm[lat_col] = pd.to_numeric(dfm[lat_col], errors="coerce")
            dfm[lon_col] = pd.to_numeric(dfm[lon_col], errors="coerce")
            dfm = dfm.dropna(subset=[lat_col, lon_col])
            if not dfm.empty:
                layer_all = pdk.Layer(
                    "ScatterplotLayer", data=dfm,
                    get_position=[lon_col, lat_col],
                    get_radius=600, get_fill_color=[200, 0, 0, 200],
                    pickable=True, radius_min_pixels=4, radius_max_pixels=20,
                )
                layers = [layer_all]

                if st.session_state.selected_comune:
                    sel = _norm_text(st.session_state.selected_comune)
                    df_sel = dfm[dfm["comune"].astype(str).map(_norm_text).eq(sel)]
                    if not df_sel.empty:
                        layer_sel = pdk.Layer(
                            "ScatterplotLayer", data=df_sel,
                            get_position=[lon_col, lat_col],
                            get_radius=900, get_fill_color=[255, 120, 0, 230],
                            pickable=True, radius_min_pixels=6, radius_max_pixels=24,
                        )
                        layers.append(layer_sel)
                        center_lat = float(df_sel[lat_col].mean())
                        center_lon = float(df_sel[lon_col].mean())
                        zoom = _zoom_for_bbox(df_sel, lat_col, lon_col)
                    else:
                        center_lat, center_lon, zoom = MILANO_LAT, MILANO_LON, 11.0
                else:
                    center_lat, center_lon, zoom = MILANO_LAT, MILANO_LON, 11.0

                view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0, bearing=0)
                tooltip = {"html": "<b>{comune}</b><br/>{ospedale}<br/>{indirizzo}<br/>{telefono}",
                           "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}}
                st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip), use_container_width=True)
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
        form_d = build_form_dict(
            gender=gender, age=age, highbp=highbp, highchol=highchol, cholcheck=cholcheck,
            smoker=smoker, stroke=stroke, heartdisease=heartdisease, physactivity=physactivity,
            fruits=fruits, veggies=veggies, hvyalcoh=hvyalcoh, anyhealthcare=anyhealthcare,
            nomedicalcare=nomedicalcare, genhlth=genhlth, menthlth=menthlth, physhlth=physhlth,
            diffwalk=diffwalk, education=education, income=income, peso=peso, altezza_cm=h
        )
        st.session_state.last_form = form_d

        rec = pd.DataFrame([form_d])

        try:
            model, model_type, meta = load_best_model()
            X = preprocess_for_inference(rec, meta)
            # pred + prob
            if model_type == "sklearn":
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(X)[0]; cls = int(np.argmax(p)); prob = float(p[cls])
                elif hasattr(model, "decision_function"):
                    s = model.decision_function(X)
                    if s.ndim == 1:
                        p1 = 1/(1+np.exp(-s[0])); cls = int(p1 >= .5); prob = float(p1 if cls else 1-p1)
                    else:
                        p = np.exp(s[0]-np.max(s[0])); p/=p.sum(); cls = int(np.argmax(p)); prob = float(p[cls])
                else:
                    cls = int(model.predict(X)[0]); prob = 0.50
            else:
                p = model.predict(X, verbose=0)[0]; cls = int(np.argmax(p)); prob = float(p[cls])
        except Exception as e:
            st.error(f"Errore durante la predizione: {e}")
            return

        st.session_state.last_pred = cls
        st.session_state.last_prob = prob

        try:
            log_prediction(form_d, cls, prob)
        except Exception:
            pass

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

    # Consiglio + tasto contatti
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
