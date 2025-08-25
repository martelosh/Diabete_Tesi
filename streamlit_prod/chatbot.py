from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import os
from pathlib import Path
import numpy as np

# PDF -> testo
try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    from PyPDF2 import PdfReader  # fallback

# TF-IDF per il retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === DeepSeek client (legge la chiave dal .env) ===
from dotenv import load_dotenv
from openai import OpenAI

# Root del progetto (cartella repo)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY  = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY mancante nel .env")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === Percorsi ===
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "CS-PANORAMA-DIABETE-LANCIO-DEF.pdf"

# === Indice TF-IDF del PDF ===
_vectorizer: Optional[TfidfVectorizer] = None
_doc_matrix = None
_chunks: List[str] = []

def _pdf_to_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    text = text.replace("\r", " ").replace("\t", " ")
    words = text.split()
    chunks, cur, cur_len = [], [], 0
    for w in words:
        cur.append(w); cur_len += len(w) + 1
        if cur_len >= max_chars:
            whole = " ".join(cur)
            chunks.append(whole)
            # overlap “grezzo” (va bene per un TF-IDF veloce)
            back = whole[-overlap:]
            cur = [back]; cur_len = len(back)
    if cur:
        chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if len(c.strip()) > 30]

def build_pdf_index(force: bool = False) -> Tuple[bool, str]:
    """Costruisce (o ricostruisce) l'indice TF-IDF del PDF."""
    global _vectorizer, _doc_matrix, _chunks
    if _vectorizer is not None and not force:
        return True, "Indice già pronto."

    raw = _pdf_to_text(PDF_PATH)
    if not raw.strip():
        _vectorizer, _doc_matrix, _chunks = None, None, []
        return False, f"PDF non trovato o vuoto: {PDF_PATH}"

    _chunks = _chunk_text(raw, max_chars=900, overlap=150)
    if not _chunks:
        _vectorizer, _doc_matrix = None, None
        return False, "Nessun contenuto indicizzabile nel PDF."

    # Nota: scikit-learn supporta 'english' come stop_words stringa. Per italiano usa None o una lista personalizzata.
    _vectorizer = TfidfVectorizer(stop_words=None)
    _doc_matrix = _vectorizer.fit_transform(_chunks)
    return True, f"Indicizzate {len(_chunks)} sezioni dal PDF."

def _retrieve(query: str, k: int = 4) -> List[str]:
    """Ritorna i top-k chunk più affini alla query."""
    if _vectorizer is None or _doc_matrix is None or not _chunks:
        ok, _ = build_pdf_index()
        if not ok:
            return []
    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _doc_matrix)[0]
    idx = np.argsort(sims)[::-1][:k]
    return [_chunks[i] for i in idx if sims[i] > 0.01]

def _deepseek_chat(messages: List[Dict[str, str]], temperature: float = 0.6, max_tokens: int = 600) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    return resp.choices[0].message.content

# === API pubblica: risposta con RAG sul PDF + FAQ del sito ===
def answer_with_rag(user_question: str, site_faq: str = "") -> str:
    """
    - Recupera i migliori estratti dal PDF e li passa come contesto.
    - Facoltativamente aggiunge un breve 'site_faq' per rispondere a come usare il sito.
    """
    contexts = _retrieve(user_question, k=4)
    context_block = "\n\n".join([f"- {c}" for c in contexts]) if contexts else "Nessun estratto utile trovato."

    system = (
        "Sei un assistente prudente. Non fornisci diagnosi, inviti a rivolgersi a medici.\n"
        "Usa il contesto seguente per rispondere in modo conciso e concreto. Se non trovi la risposta, dillo chiaramente."
    )
    if site_faq:
        system += f"\n\n[FAQ Sito]\n{site_faq}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Domanda: {user_question}\n\n[Contesto dal PDF]\n{context_block}"}
    ]
    try:
        return _deepseek_chat(messages)
    except Exception as e:
        return f"⚠️ Errore nel contattare il modello: {e}"

# Costruisce l’indice al primo import
build_pdf_index(force=True)
