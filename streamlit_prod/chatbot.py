from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# PDF -> testo e indicizzazione
try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    from PyPDF2 import PdfReader  # fallback

from sklearn.feature_extraction.text import TfidfVectorizer  # scikit-learn già nel progetto
from sklearn.metrics.pairwise import cosine_similarity

# DeepSeek via client OpenAI-compatibile
from openai import OpenAI  # pip install openai

# === CONFIG API (al momento come da tua richiesta, in chiaro) ===
DEEPSEEK_API_KEY  = 'sk-b5aec58aa1384eaf8e1b769b646adb58'
DEEPSEEK_MODEL   = "deepseek-chat"            # oppure "deepseek-reasoner"
DEEPSEEK_BASEURL = "https://api.deepseek.com"

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "CS-PANORAMA-DIABETE-LANCIO-DEF.pdf"

# === CLIENT ===
_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASEURL)

# === INDICE PDF (TF-IDF) ===
_vectorizer: Optional[TfidfVectorizer] = None
_doc_matrix = None
_chunks: List[str] = []

def _pdf_to_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        return ""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    text = text.replace("\r", " ").replace("\t", " ")
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= max_chars:
            chunks.append(" ".join(cur))
            # overlap
            back = " ".join(cur)[-overlap:]
            cur = [back]
            cur_len = len(back)
    if cur:
        chunks.append(" ".join(cur))
    # filtra chunk troppo corti o vuoti
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
    return chunks

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

    # lingua italiana -> stopwords italiane se disponibile
    _vectorizer = TfidfVectorizer(stop_words="italian")
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
    resp = _client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    return resp.choices[0].message.content

# === API PUBBLICA ===
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

# costruisci l'indice una volta all'import
build_pdf_index(force=True)