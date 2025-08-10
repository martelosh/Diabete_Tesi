# src/chatbot.py
import os
import json
import difflib
import unicodedata
import requests
import pandas as pd
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Puoi cambiare qui il nome del file; se esiste la ENV, usa quella
HOSPITALS_CSV = os.getenv(
    "HOSPITALS_CSV",
    str(PROJECT_ROOT / "data" / "ospedali_milano_comuni_mapping.csv")
)

def deepseek_chat(messages, system_prompt=None, temperature=0.3):
    """Wrapper DeepSeek con gestione errori 'gentile'."""
    if not DEEPSEEK_API_KEY:
        return "⚠️ Chatbot non disponibile: API key mancante."

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 402:
            return "⚠️ Chatbot temporaneamente non attivo (credito o piano non abilitato)."
        return f"⚠️ Errore chatbot ({status}): riprova più tardi."
    except Exception:
        return "⚠️ Chatbot non raggiungibile al momento."

def _normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.strip().lower()

@lru_cache(maxsize=1)
def _load_contacts_df() -> pd.DataFrame | None:
    """Carica e normalizza il CSV degli ospedali/comuni."""
    path = Path(HOSPITALS_CSV)
    if not path.exists():
        return None
    df = pd.read_csv(path)

    # Mappa nomi colonne più comuni → standard
    colmap_options = {
        "comune": ["comune", "Comune", "COMUNE", "citta", "città", "municipio"],
        "struttura": ["ospedale", "struttura", "nome_struttura", "denominazione"],
        "indirizzo": ["indirizzo", "via", "address", "ind"],
        "telefono": ["telefono", "tel", "numero", "phone", "recapito"],
        "note": ["note", "descrizione", "info"],
        "distanza": ["distanza", "km", "distance_km"]
    }

    def pick_col(target):
        for opt in colmap_options[target]:
            if opt in df.columns:
                return opt
        return None

    c_comune   = pick_col("comune")
    c_strutt   = pick_col("struttura")
    c_addr     = pick_col("indirizzo")
    c_phone    = pick_col("telefono")
    c_note     = pick_col("note")
    c_dist     = pick_col("distanza")

    # Colonne minime richieste
    if not c_comune:
        # se manca 'comune' non possiamo fare match
        return None

    # Costruiamo colonne standard (con fallback su stringhe vuote)
    out = pd.DataFrame({
        "comune":   df[c_comune].astype(str),
        "struttura": df[c_strutt].astype(str) if c_strutt else "",
        "indirizzo": df[c_addr].astype(str) if c_addr else "",
        "telefono":  df[c_phone].astype(str) if c_phone else "",
        "note":      df[c_note].astype(str) if c_note else "",
        "distanza":  df[c_dist] if c_dist else None
    })

    # Colonna normalizzata per il match
    out["_comune_norm"] = out["comune"].map(_normalize)
    return out

def get_nearby_contacts(comune: str, max_results: int = 3):
    """
    Ritorna una lista di dict: {struttura, indirizzo, telefono, tipo}
    - legge da data/ospedali_milano_comuni_mapping.csv (o env HOSPITALS_CSV)
    - match case-insensitive e accent-insensitive
    - fallback su fuzzy matching se non trova match esatto
    """
    df = _load_contacts_df()
    if df is None or not isinstance(comune, str) or not comune.strip():
        # fallback molto generico
        return [{"struttura": "Medico di base / CUP regionale", "indirizzo": "", "telefono": "Numero verde regionale", "tipo": "info/prenotazioni"}]

    q = _normalize(comune)
    # match diretto
    exact = df[df["_comune_norm"] == q]
    if not exact.empty:
        take = exact.head(max_results)
    else:
        # fuzzy: trova similari sul set dei comuni
        all_names = df["_comune_norm"].dropna().unique().tolist()
        best = difflib.get_close_matches(q, all_names, n=max_results, cutoff=0.6)
        take = df[df["_comune_norm"].isin(best)].head(max_results)

    results = []
    for _, r in take.iterrows():
        results.append({
            "struttura": r.get("struttura") or "Struttura sanitaria",
            "indirizzo": r.get("indirizzo") or "",
            "telefono":  r.get("telefono") or "",
            "tipo":      (r.get("note") or "prenotazioni/contatti")
        })
    if not results:
        results = [{"struttura": "Medico di base / CUP regionale", "indirizzo": "", "telefono": "Numero verde regionale", "tipo": "info/prenotazioni"}]
    return results
