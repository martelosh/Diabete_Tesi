import os
import json
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def deepseek_chat(messages, system_prompt=None, temperature=0.3):
    if not DEEPSEEK_API_KEY:
        return "⚠️ API key DeepSeek mancante. Configura DEEPSEEK_API_KEY nel file .env."

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
    except Exception as e:
        return f"Errore nella chiamata al modello: {e}"

def get_nearby_contacts(comune: str):
    """
    Cerca contatti predefiniti:
     - Policlinico Gemelli (Roma) [contact centric]
     - CUP Piemonte (regionale)
     - Pronto Diabete (numero verde)
    Se il comune contiene “rom”, usiamo Gemelli; se “piemonte”, CUP; altrimenti fallback.
    """
    comune_low = comune.strip().lower() if comune else ""
    results = []

    if "rom" in comune_low:
        results = [
            {"struttura": "Policlinico Gemelli - Diabetologia", "telefono": "06 3055.612 (CUP)", "tipo": "prenotazioni"},
        ]
    elif "piem" in comune_low or "torin" in comune_low or "novar" in comune_low:
        results = [
            {"struttura": "CUP Piemonte (call center)", "telefono": "800-000 500", "tipo": "prenotazioni CUP regionale"},
        ]
    else:
        results = [
            {"struttura": "Pronto Diabete (numero verde)", "telefono": "800-042 747", "tipo": "consulenze/prevenzione"},
        ]

    return results
