import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # <-- root progetto
sys.path.append(str(PROJECT_ROOT)) 

st.set_page_config(page_title="Valutazione Rischio Diabete", page_icon="🩺", layout="wide")
st.markdown("""<style>
section[data-testid="stSidebar"]{display:none}header [data-testid="baseButton-headerNoPadding"]{visibility:hidden}
div.block-container{padding:2rem 0}
.hero{border-radius:28px;padding:clamp(1.6rem,2.5vw,2.6rem);border:1px solid rgba(0,0,0,.06);
background:radial-gradient(1200px 600px at 8% 10%, rgba(0,120,255,.10), transparent 60%),
radial-gradient(1000px 500px at 90% 30%, rgba(255,60,140,.10), transparent 60%),
linear-gradient(180deg, rgba(255,255,255,.94), rgba(255,255,255,.88));box-shadow:0 14px 44px rgba(0,0,0,.08)}
#floating-chat{position:fixed;right:24px;bottom:24px;z-index:9999}
#floating-chat button{border-radius:999px;padding:14px 16px;font-weight:800;border:1px solid rgba(0,0,0,.1);background:#fff;box-shadow:0 14px 30px rgba(0,0,0,.15);cursor:pointer}
.chat-panel{position:fixed;right:24px;bottom:90px;z-index:9998;width:min(440px,94vw);max-height:72vh;overflow:auto;background:#fff;
border:1px solid rgba(0,0,0,.10);border-radius:18px;box-shadow:0 22px 52px rgba(0,0,0,.22);padding:.9rem}
#chat-teaser{position:fixed;right:88px;bottom:110px;z-index:9999;max-width:min(380px,72vw);background:#fff;color:#111;
border:1px solid rgba(0,0,0,.12);border-radius:14px;box-shadow:0 16px 40px rgba(0,0,0,.18);padding:10px 12px;font-size:.95rem;line-height:1.35}
#chat-teaser:after{content:"";position:absolute;right:-10px;bottom:14px;border:10px solid;border-color:transparent transparent transparent #fff}
.teaser-actions{display:flex;gap:8px;margin-top:6px;justify-content:flex-end}
@media (prefers-color-scheme: dark){.hero{border-color:rgba(255,255,255,.08);background:linear-gradient(180deg,#0f1117,#0f121a);color:#e9e9ea}
.chat-panel,#floating-chat button,#chat-teaser{background:#101318;color:#e9e9ea;border-color:rgba(255,255,255,.08)}
#chat-teaser:after{border-color:transparent transparent transparent #101318}}
</style>""", unsafe_allow_html=True)

# -------- helper locali (no LLM) --------
CONTACTS_CSV = PROJECT_ROOT / "data" / "ospedali_milano_comuni_mapping.csv"

def get_nearby_contacts(comune: str, top: int = 5):
    if not CONTACTS_CSV.exists():
        return []
    df = pd.read_csv(CONTACTS_CSV)
    df.columns = [c.lower() for c in df.columns]
    if "comune" not in df.columns:
        return []
    m = df[df["comune"].astype(str).str.contains(comune, case=False, na=False)].head(top)
    return m.to_dict(orient="records")

def make_teaser(pred: int, prob: float) -> str:
    label = {0: "nessun diabete", 1: "pre-diabete", 2: "diabete"}.get(int(pred), "")
    return (
        f"Risultato stimato: **{pred}** ({label}) — probabilità **{prob*100:.1f}%**.\n"
        "Questo non sostituisce un parere medico.\n\n"
        "Scrivimi il tuo **comune** per mostrarti i contatti utili e aiutarti a prenotare."
    )

# -------- stato condiviso chat --------
for k, v in {"last_pred":None, "last_prob":None, "messages":[], "chat_open":False,
             "teaser_message":None, "show_teaser":False}.items():
    st.session_state.setdefault(k, v)

# -------- HOME --------
st.markdown("""<div class="hero"><h1>🩺 Valutazione del Rischio Diabete</h1>
<p class="small">Compila il form: ottieni classe (0/1/2) e <b>probabilità</b>. L’assistente ti aiuta a prenotare.</p></div>""",
            unsafe_allow_html=True)
st.write("")
c1, c2, c3 = st.columns(3)
with c1: st.markdown("### 🎯 Perché\n- Screening rapido\n- Probabilità oltre la classe\n- Supporto prenotazioni")
with c2: st.markdown("### 🔒 Privacy\nUso limitato a questa valutazione.\nNon sostituisce un medico.")
with c3: st.markdown("### ⚙️ Passi\n1) Form\n2) Risultato+prob\n3) Chat")

if st.button("📝 Apri il form", type="primary", use_container_width=True):
    try: st.switch_page("pages/1_Form.py")
    except Exception: st.info("Apri il form dal menu delle pagine.")

# -------- CHAT floating (anche da Home) --------
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
        if st.session_state.last_pred is None:
            st.session_state.update(teaser_message="Prima compila il form.", show_teaser=True, chat_open=False)
        else:
            st.session_state.update(chat_open=True, show_teaser=False)
        st.query_params.clear()
    if qp.get("teaser") == "close":
        st.session_state.show_teaser = False; st.query_params.clear()

    # teaser
    if st.session_state.show_teaser and not st.session_state.chat_open and st.session_state.teaser_message:
        text = st.session_state.teaser_message
        if len(text) > 240: text = text[:235].rstrip() + "…"
        actions = ('<button type="submit" name="chat" value="open">Apri chat</button>'
                   '<button type="submit" name="teaser" value="close">Chiudi</button>') \
                  if st.session_state.last_pred is not None else \
                  ('<button type="submit" name="chat" value="toggle">Compila il form</button>'
                   '<button type="submit" name="teaser" value="close">Chiudi</button>')
        st.markdown(f'<div id="chat-teaser"><div>{text}</div><div class="teaser-actions"><form>{actions}</form></div></div>',
                    unsafe_allow_html=True)

    # pannello chat
    if st.session_state.chat_open and st.session_state.last_pred is not None:
        st.markdown('<div class="chat-panel">**Assistente**  \n<small>Non sostituisce un consulto medico.</small><br/><br/>',
                    unsafe_allow_html=True)
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Scrivi un messaggio…"):
            st.session_state.messages.append({"role":"user","content":prompt})
            reply = None
            comune = prompt.strip()
            if len(comune) >= 2 and any(c.isalpha() for c in comune):
                contacts = get_nearby_contacts(comune)
                if contacts:
                    rows = []
                    for c in contacts:
                        c = {k.lower(): v for k, v in c.items()}  # normalizza chiavi
                        r = f"- **{c.get('struttura','Struttura')}**"
                        if c.get("indirizzo"): r += f" — {c['indirizzo']}"
                        if c.get("telefono"):  r += f" — 📞 {c['telefono']}"
                        if c.get("tipo"):      r += f"  \n  _{c['tipo']}_"
                        rows.append(r)
                    reply = "**Contatti utili nella tua zona**:\n" + "\n".join(rows)

            if not reply:
                reply = make_teaser(st.session_state.last_pred or 0, st.session_state.last_prob or 0.0)

            st.session_state.messages.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"): st.markdown(reply)

        st.markdown('</div>', unsafe_allow_html=True)

render_chat()