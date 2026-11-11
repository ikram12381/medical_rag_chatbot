"""
Interface web Streamlit — Assistant médical marocain (RAG hybride FAISS + BM25).

Fonctionnalités :
- Conversations multiples (création, bascule, suppression, export)
- Réponses en flux, citations [Source N] rendues en pastilles
- Cartes de sources avec score de pertinence et extrait du passage
- Indicateurs par réponse : langue, mode de recherche, latence, garde-fou
- Retour utilisateur (pouce haut / bas) par réponse
- Tableau de bord de session (latence, pertinence, usage des sources)
- Vérification de l'état du système (index, clé API)

Lancer avec :
    streamlit run app/streamlit_app.py
"""
import html
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from config import API_KEY, INDEX_DIR, LOW_RELEVANCE_THRESHOLD
from src.rag.pipeline import ask

st.set_page_config(
    page_title="Assistant Médical Maroc",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Constantes
# ============================================================
SUGGESTIONS = [
    ("Diabète", "Quels sont les symptômes du diabète type 2 ?"),
    ("Hypertension", "Comment prévenir les complications de l'hypertension artérielle ?"),
    ("Vaccination", "Quel est le calendrier national de vaccination au Maroc ?"),
    ("Rougeole", "Comment isoler un cas de rougeole ?"),
    ("Pied diabétique", "Comment prendre soin de ses pieds quand on est diabétique ?"),
    ("Mode de vie", "Quels sont les modes de vie sains recommandés pour un hypertendu ?"),
]

# Exemples en darija (écriture latine) : le titre est en français pour rester lisible.
DARIJA_SUGGESTIONS = [
    ("Diabète", "chno homa l-3alamat dyal diabète type 2 ?"),
    ("Vaccination", "wach l-vaccin dyal l-rougeole darouri l-drari ?"),
    ("Hypertension", "kifach n7mi rasi mn l-tension l-3alya ?"),
]

EMERGENCY_NUMBERS = [("SAMU", "141"), ("Protection civile", "15"), ("Police secours", "19")]

USER_AVATAR = ":material/person:"
DISCLAIMER_SEPARATOR = "\n\n---\n"
CITATION_RE = re.compile(r"\[Source (\d+)\]")


# ============================================================
# Thème (CSS)
# ============================================================
def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
    --ink: #0B2540;
    --primary: #0F4C81;
    --primary-soft: #E7F0F8;
    --teal: #0E8F83;
    --teal-soft: #E3F4F1;
    --amber: #B7791F;
    --amber-soft: #FDF4E3;
    --red: #B42318;
    --red-soft: #FDECEA;
    --bg: #F5F8FB;
    --surface: #FFFFFF;
    --line: #DCE5EE;
    --muted: #5F6F82;
    --radius: 12px;
}

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: var(--bg); color: var(--ink); }
.block-container { max-width: 1080px; padding-top: 1.6rem; padding-bottom: 6rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.brand { display: flex; align-items: center; gap: .7rem; margin-bottom: 1.1rem; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(140deg, var(--primary), var(--teal));
    display: grid; place-items: center; color: #fff; font-size: 1.15rem;
}
.brand-name { font-weight: 700; color: var(--ink); line-height: 1.15; font-size: 1rem; }
.brand-sub { color: var(--muted); font-size: .75rem; }
.side-label { color: var(--muted); font-size: .78rem; font-weight: 600; margin: 1rem 0 .35rem; }

.status-row { display: flex; align-items: center; gap: .5rem; font-size: .82rem; padding: .22rem 0; color: var(--ink); }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dot-ok { background: var(--teal); box-shadow: 0 0 0 3px var(--teal-soft); }
.dot-ko { background: var(--red); box-shadow: 0 0 0 3px var(--red-soft); }

.emergency {
    border: 1px solid #F3C9C4; background: var(--red-soft);
    border-radius: var(--radius); padding: .7rem .85rem; margin-top: 1rem;
}
.emergency-title { color: var(--red); font-weight: 700; font-size: .85rem; margin-bottom: .3rem; }
.emergency-row { display: flex; justify-content: space-between; font-size: .83rem; color: #5B1A14; padding: .08rem 0; }
.emergency-row b { font-variant-numeric: tabular-nums; }

/* ---------- Hero ---------- */
.hero { padding: 1.2rem 0 .4rem; }
.hero h1 {
    font-family: 'Newsreader', Georgia, serif; font-weight: 600;
    font-size: 2.35rem; line-height: 1.15; color: var(--ink);
    margin: 0 0 .5rem; letter-spacing: -.01em; padding: 0;
}
.hero p { color: var(--muted); max-width: 58ch; font-size: 1.02rem; line-height: 1.55; margin: 0; }
.pills { display: flex; flex-wrap: wrap; gap: .45rem; margin: 1rem 0 1.4rem; }
.pill {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .28rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
    background: var(--primary-soft); color: var(--primary); border: 1px solid #C9DCEC;
}
.pill-teal { background: var(--teal-soft); color: var(--teal); border-color: #BFE5DF; }
.pill-amber { background: var(--amber-soft); color: var(--amber); border-color: #F0DBB0; }
.pill-red { background: var(--red-soft); color: var(--red); border-color: #F3C9C4; }
.pill-grey { background: #EEF2F6; color: var(--muted); border-color: var(--line); }

/* Boutons */
.stButton > button {
    border-radius: var(--radius); border: 1px solid var(--line);
    background: var(--surface); color: var(--ink); font-weight: 500;
    transition: border-color .15s, box-shadow .15s;
}
.stButton > button:hover { border-color: var(--primary); color: var(--primary); box-shadow: 0 2px 10px rgba(15,76,129,.08); }
.stButton > button:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: var(--primary); color: #fff; border-color: var(--primary); }
.stButton > button[kind="primary"]:hover { background: #0C3E6A; color: #fff; }

/* Cartes de suggestions */
[class*="st-key-sg-"] button {
    height: 100%; min-height: 6.4rem; padding: .85rem 1rem;
    justify-content: flex-start; align-items: flex-start; text-align: left;
}
[class*="st-key-sg-"] button > div { width: 100%; justify-content: flex-start; }
[class*="st-key-sg-"] { height: 100%; }
[class*="st-key-sg-"] button p { width: 100%; white-space: normal; text-align: left; line-height: 1.45; font-size: .88rem; color: var(--muted); }
[class*="st-key-sg-"] button p strong { display: block; color: var(--ink); font-size: .95rem; margin-bottom: .2rem; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 1rem 1.1rem; margin-bottom: .8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--primary-soft); border-color: #C9DCEC;
}
[data-testid="stChatMessage"] p { line-height: 1.65; }
[data-testid="stChatInput"] { border-radius: var(--radius); border-color: var(--line); }

.cite {
    display: inline-block; min-width: 1.25rem; text-align: center;
    padding: 0 .3rem; margin: 0 .1rem; border-radius: 6px;
    background: var(--primary-soft); color: var(--primary);
    font-size: .72rem; font-weight: 700; vertical-align: super; line-height: 1.35;
    border: 1px solid #C9DCEC;
}
.meta { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .6rem; }
.note {
    border-left: 3px solid var(--amber); background: var(--amber-soft);
    padding: .6rem .8rem; border-radius: 8px; color: #6B4A12; font-size: .85rem; margin-top: .8rem;
}
.note-guard { border-left-color: var(--teal); background: var(--teal-soft); color: #0B5F57; margin-top: 0; }
.translated { color: var(--muted); font-size: .85rem; margin-bottom: .4rem; }

/* ---------- Cartes de sources ---------- */
.src-card {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: var(--radius); padding: .8rem .95rem; margin-bottom: .6rem;
}
.src-head { display: flex; align-items: center; gap: .6rem; }
.src-num {
    width: 24px; height: 24px; border-radius: 7px; background: var(--primary);
    color: #fff; font-size: .78rem; font-weight: 700; display: grid; place-items: center; flex: none;
}
.src-name { font-weight: 600; font-size: .92rem; color: var(--ink); flex: 1; overflow-wrap: anywhere; }
.src-score { font-size: .78rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.bar { height: 5px; border-radius: 99px; background: #E8EEF4; margin: .55rem 0 .5rem; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 99px; }
.bar-high > span { background: var(--teal); }
.bar-mid > span { background: var(--primary); }
.bar-low > span { background: #9AA8B8; }
.src-excerpt { color: #3E4C5E; font-size: .84rem; line-height: 1.55; }
.src-full { margin-top: .5rem; }
.src-full summary { cursor: pointer; color: var(--primary); font-size: .8rem; font-weight: 600; }
.src-full summary:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }
.src-full .src-excerpt { margin-top: .4rem; padding-top: .5rem; border-top: 1px solid var(--line); }

/* ---------- Dashboard ---------- */
.kpi { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: .9rem 1rem; }
.kpi-v { font-size: 1.7rem; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; line-height: 1.1; }
.kpi-l { color: var(--muted); font-size: .8rem; margin-top: .2rem; }

h2, h3 { color: var(--ink) !important; letter-spacing: -.005em; }
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: var(--radius) !important; background: var(--surface); }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
@media (max-width: 640px) { .hero h1 { font-size: 1.7rem; } }
</style>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# Utilitaires
# ============================================================
def esc(text: str) -> str:
    return html.escape(str(text))


def source_score(s: dict) -> float:
    return float(s.get("similarity_score", s.get("fused_score", 0.0)))


def score_tier(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.35:
        return "mid"
    return "low"


def pretty_name(doc: str) -> str:
    return doc.replace("_", " ").replace("-", " ").strip().capitalize()


def split_disclaimer(answer: str) -> tuple[str, str]:
    """Sépare le corps de la réponse du disclaimer ajouté par le pipeline."""
    if DISCLAIMER_SEPARATOR in answer:
        body, _, tail = answer.rpartition(DISCLAIMER_SEPARATOR)
        return body.strip(), tail.strip().strip("*⚠️ ").strip()
    return answer.strip(), ""


def render_citations(text: str) -> str:
    """Remplace [Source N] par des pastilles numérotées."""
    return CITATION_RE.sub(lambda m: f'<span class="cite" title="Source {m.group(1)}">{m.group(1)}</span>', text)


def index_status() -> dict:
    return {
        "Index vectoriel (FAISS)": (INDEX_DIR / "faiss.index").exists(),
        "Index lexical (BM25)": (INDEX_DIR / "bm25.pkl").exists(),
        "Clé API configurée": bool(API_KEY),
    }


# ============================================================
# État de session
# ============================================================
def new_conversation() -> str:
    cid = uuid.uuid4().hex[:8]
    st.session_state.convs[cid] = {"title": "Nouvelle conversation", "history": [], "created": datetime.now()}
    st.session_state.active = cid
    return cid


def init_state() -> None:
    if "convs" not in st.session_state:
        st.session_state.convs = {}
        new_conversation()
    st.session_state.setdefault("view", "Assistant")
    st.session_state.setdefault("pending", None)


def active_conv() -> dict:
    return st.session_state.convs[st.session_state.active]


def all_entries() -> list[dict]:
    return [e for c in st.session_state.convs.values() for e in c["history"]]


# ============================================================
# Composants de rendu
# ============================================================
def render_meta(entry: dict) -> None:
    pills = []
    if entry["rejected"]:
        pills.append('<span class="pill pill-amber">Hors périmètre</span>')
    else:
        pills.append(f'<span class="pill pill-teal">{len(entry["sources"])} source(s)</span>')
    pills.append(f'<span class="pill">{"Darija" if entry["darija"] else "Français"}</span>')
    pills.append(f'<span class="pill pill-grey">{"Recherche hybride" if entry["hybrid"] else "Sémantique seule"}</span>')
    pills.append(f'<span class="pill pill-grey">{entry["latency"]:.1f} s</span>')
    st.markdown(f'<div class="meta">{"".join(pills)}</div>', unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("Aucune source n'a été retrouvée pour cette question.")
        return
    for i, s in enumerate(sources, start=1):
        score = source_score(s)
        tier = score_tier(score)
        width = max(4, min(100, int(score * 100)))
        text = s.get("text", "").strip().replace("\n", " ")
        excerpt = text[:280] + ("…" if len(text) > 280 else "")
        bm25 = f' · BM25 {s["bm25_score"]:.1f}' if "bm25_score" in s else ""
        full_html = (
            f'<details class="src-full"><summary>Lire le passage complet</summary>'
            f'<div class="src-excerpt">{esc(s.get("text", "")).replace(chr(10), "<br>")}</div></details>'
            if len(text) > 280 else ""
        )
        st.markdown(
            f"""
<div class="src-card">
  <div class="src-head">
    <div class="src-num">{i}</div>
    <div class="src-name">{esc(pretty_name(s["source_document"]))}</div>
    <div class="src-score">{score:.2f}<span style="color:var(--muted);font-weight:500">{bm25}</span></div>
  </div>
  <div class="bar bar-{tier}"><span style="width:{width}%"></span></div>
  <div class="src-excerpt">{esc(excerpt)}</div>
  {full_html}
</div>""",
            unsafe_allow_html=True,
        )


def render_answer_body(entry: dict) -> None:
    body, disclaimer = split_disclaimer(entry["answer"])
    if entry["darija"] and entry["question_fr"] and entry["question_fr"] != entry["question"]:
        st.markdown(
            f'<div class="translated">Question comprise : « {esc(entry["question_fr"])} »</div>',
            unsafe_allow_html=True,
        )
    if entry["rejected"]:
        st.markdown(f'<div class="note note-guard">{esc(body)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(render_citations(body), unsafe_allow_html=True)
    if disclaimer:
        st.markdown(f'<div class="note">{esc(disclaimer)}</div>', unsafe_allow_html=True)
    render_meta(entry)


def render_entry(entry: dict, conv_id: str, idx: int) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(entry["question"])
    with st.chat_message("assistant", avatar="🩺"):
        render_answer_body(entry)
        if entry["sources"]:
            with st.expander(f"Sources consultées ({len(entry['sources'])})"):
                render_sources(entry["sources"])
        value = st.feedback("thumbs", key=f"fb-{conv_id}-{idx}")
        if value is not None:
            entry["feedback"] = "up" if value == 1 else "down"


def stream_words(text: str):
    for token in re.split(r"(\s+)", text):
        yield token
        time.sleep(0.012)


def run_question(question: str, respond_in_darija: bool, top_k: int, use_hybrid: bool) -> None:
    conv = active_conv()
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(question)
    with st.chat_message("assistant", avatar="🩺"):
        t0 = time.perf_counter()
        try:
            with st.status("Recherche dans les sources médicales…", expanded=False) as status:
                result = ask(question, respond_in_darija=respond_in_darija, top_k=top_k, use_hybrid=use_hybrid)
                latency = time.perf_counter() - t0
                status.update(label=f"Réponse prête en {latency:.1f} s", state="complete")
        except FileNotFoundError:
            st.error("L'index vectoriel n'existe pas encore. Lancez : `python -m src.indexing.vector_store`")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"La requête a échoué : {exc}. Vérifiez la clé API et la connexion réseau.")
            return

        entry = {
            "question": question,
            "question_fr": result.question_fr,
            "answer": result.answer,
            "sources": result.sources,
            "darija": result.input_was_darija,
            "rejected": result.was_rejected_low_relevance,
            "hybrid": use_hybrid,
            "latency": latency,
            "time": datetime.now(),
            "feedback": None,
        }
        body, _ = split_disclaimer(entry["answer"])
        if not entry["rejected"]:
            st.write_stream(stream_words(body))
    conv["history"].append(entry)
    if conv["title"] == "Nouvelle conversation":
        conv["title"] = question[:42] + ("…" if len(question) > 42 else "")
    st.rerun()


# ============================================================
# Exports
# ============================================================
def export_markdown(conv: dict) -> str:
    lines = [f"# {conv['title']}", f"_Exporté le {datetime.now():%d/%m/%Y %H:%M}_", ""]
    for e in conv["history"]:
        body, _ = split_disclaimer(e["answer"])
        lines += [f"## Question\n{e['question']}", "", f"**Réponse**\n\n{body}", ""]
        if e["sources"]:
            lines.append("**Sources**")
            lines += [f"- {pretty_name(s['source_document'])} (score {source_score(s):.2f})" for s in e["sources"]]
            lines.append("")
    lines.append("> Information générale uniquement. Elle ne remplace pas une consultation médicale.")
    return "\n".join(lines)


def export_json(conv: dict) -> str:
    def default(o):
        return o.isoformat() if isinstance(o, datetime) else str(o)

    return json.dumps(conv, ensure_ascii=False, indent=2, default=default)


# ============================================================
# Barre latérale
# ============================================================
VIEWS = ["Assistant", "Tableau de bord", "À propos"]


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">🩺</div>'
            '<div><div class="brand-name">Assistant médical</div>'
            '<div class="brand-sub">Sources officielles du Maroc</div></div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Nouvelle conversation", type="primary", width="stretch"):
            new_conversation()
            st.session_state.view = "Assistant"
            st.rerun()

        st.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
        st.session_state.view = st.radio(
            "Navigation", VIEWS, index=VIEWS.index(st.session_state.view), label_visibility="collapsed"
        )

        st.markdown('<div class="side-label">Conversations</div>', unsafe_allow_html=True)
        for cid, conv in reversed(list(st.session_state.convs.items())):
            is_active = cid == st.session_state.active
            label = ("● " if is_active else "") + conv["title"]
            c1, c2 = st.columns([5, 1])
            if c1.button(label, key=f"conv-{cid}", width="stretch"):
                st.session_state.active = cid
                st.session_state.view = "Assistant"
                st.rerun()
            if c2.button("", key=f"del-{cid}", icon=":material/delete:", help="Supprimer cette conversation"):
                del st.session_state.convs[cid]
                if not st.session_state.convs:
                    new_conversation()
                elif st.session_state.active == cid:
                    st.session_state.active = next(reversed(st.session_state.convs))
                st.rerun()

        st.markdown('<div class="side-label">Paramètres de recherche</div>', unsafe_allow_html=True)
        settings = {
            "respond_in_darija": st.toggle("Répondre en Darija", value=False),
            "use_hybrid": st.toggle("Recherche hybride", value=True, help="Sémantique (FAISS) + mots-clés (BM25)"),
            "top_k": st.slider("Sources consultées", 1, 10, 5),
        }

        conv = active_conv()
        if conv["history"]:
            st.markdown('<div class="side-label">Exporter la conversation</div>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            d1.download_button("Markdown", export_markdown(conv), file_name="conversation.md", width="stretch")
            d2.download_button("JSON", export_json(conv), file_name="conversation.json", width="stretch")

        st.markdown('<div class="side-label">État du système</div>', unsafe_allow_html=True)
        rows = "".join(
            f'<div class="status-row"><span class="dot {"dot-ok" if ok else "dot-ko"}"></span>{esc(name)}</div>'
            for name, ok in index_status().items()
        )
        st.markdown(rows, unsafe_allow_html=True)

        emergency = "".join(
            f'<div class="emergency-row"><span>{n}</span><b>{num}</b></div>' for n, num in EMERGENCY_NUMBERS
        )
        st.markdown(
            f'<div class="emergency"><div class="emergency-title">En cas d\'urgence</div>{emergency}</div>',
            unsafe_allow_html=True,
        )
    return settings


# ============================================================
# Vues
# ============================================================
def view_assistant(settings: dict) -> None:
    conv = active_conv()

    if not conv["history"] and not st.session_state.pending:
        st.markdown(
            """
<div class="hero">
  <h1>Posez une question de santé, obtenez une réponse sourcée.</h1>
  <p>Les réponses s'appuient uniquement sur des documents du Ministère de la Santé et de l'OMS Maroc.
  Chaque affirmation renvoie à sa source. Vous pouvez écrire en français ou en darija.</p>
</div>
<div class="pills">
  <span class="pill">Recherche hybride FAISS + BM25</span>
  <span class="pill pill-teal">Citations vérifiables</span>
  <span class="pill pill-amber">Garde-fou anti-hallucination</span>
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-label">Questions fréquentes</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (topic, q) in enumerate(SUGGESTIONS):
            with cols[i % 3]:
                if st.button(f"**{topic}**\n\n{q}", key=f"sg-{i}", width="stretch"):
                    st.session_state.pending = q
                    st.rerun()

        st.markdown('<div class="side-label">Essayer en darija</div>', unsafe_allow_html=True)
        dcols = st.columns(3)
        for i, (topic, q) in enumerate(DARIJA_SUGGESTIONS):
            with dcols[i]:
                if st.button(f"**{topic}**\n\n{q}", key=f"sg-d-{i}", width="stretch"):
                    st.session_state.pending = q
                    st.rerun()
    else:
        st.markdown(f"### {esc(conv['title'])}")
        for idx, entry in enumerate(conv["history"]):
            render_entry(entry, st.session_state.active, idx)

    typed = st.chat_input("Posez votre question médicale (français ou darija)…")
    question = typed or st.session_state.pending
    if question:
        st.session_state.pending = None
        run_question(question, settings["respond_in_darija"], settings["top_k"], settings["use_hybrid"])


def kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="kpi-v">{value}</div><div class="kpi-l">{label}</div></div>'


def view_dashboard() -> None:
    st.markdown("## Tableau de bord de la session")
    entries = all_entries()
    if not entries:
        st.info("Aucune donnée pour le moment. Posez une question dans l'onglet Assistant pour alimenter ce tableau.")
        return

    answered = [e for e in entries if not e["rejected"]]
    rejected = len(entries) - len(answered)
    avg_latency = sum(e["latency"] for e in entries) / len(entries)
    top_scores = [max((source_score(s) for s in e["sources"]), default=0.0) for e in answered]
    avg_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
    rated = [e for e in entries if e["feedback"]]
    satisfaction = (sum(e["feedback"] == "up" for e in rated) / len(rated) * 100) if rated else None

    cols = st.columns(5)
    cols[0].markdown(kpi(str(len(entries)), "Questions posées"), unsafe_allow_html=True)
    cols[1].markdown(kpi(f"{avg_latency:.1f} s", "Latence moyenne"), unsafe_allow_html=True)
    cols[2].markdown(kpi(f"{avg_top:.2f}", "Meilleur score moyen"), unsafe_allow_html=True)
    cols[3].markdown(kpi(str(rejected), "Refusées (hors périmètre)"), unsafe_allow_html=True)
    cols[4].markdown(
        kpi(f"{satisfaction:.0f} %" if satisfaction is not None else "–", "Réponses jugées utiles"),
        unsafe_allow_html=True,
    )

    st.markdown("&nbsp;")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Latence par question")
        df = pd.DataFrame(
            {"Question": [f"Q{i + 1}" for i in range(len(entries))], "Secondes": [e["latency"] for e in entries]}
        ).set_index("Question")
        st.bar_chart(df, color="#0F4C81")
    with right:
        st.markdown("#### Documents les plus sollicités")
        counts: dict[str, int] = {}
        for e in answered:
            for s in e["sources"]:
                name = pretty_name(s["source_document"])
                counts[name] = counts.get(name, 0) + 1
        if counts:
            sdf = pd.Series(counts, name="Citations").sort_values(ascending=False).head(8).to_frame()
            st.bar_chart(sdf, color="#0E8F83", horizontal=True)
        else:
            st.caption("Aucune source utilisée pour l'instant.")

    st.markdown("#### Historique détaillé")
    table = pd.DataFrame(
        {
            "Heure": [e["time"].strftime("%H:%M") for e in entries],
            "Question": [e["question"] for e in entries],
            "Langue": ["Darija" if e["darija"] else "Français" for e in entries],
            "Sources": [len(e["sources"]) for e in entries],
            "Meilleur score": [round(max((source_score(s) for s in e["sources"]), default=0.0), 2) for e in entries],
            "Latence (s)": [round(e["latency"], 1) for e in entries],
            "Statut": ["Hors périmètre" if e["rejected"] else "Répondu" for e in entries],
            "Avis": [{"up": "👍", "down": "👎"}.get(e["feedback"], "") for e in entries],
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Exporter l'historique (CSV)", table.to_csv(index=False).encode("utf-8"), "historique.csv", "text/csv"
    )


def view_about() -> None:
    st.markdown("## À propos")
    st.markdown(
        f"""
Cet assistant fournit une **information médicale générale** à partir de documents officiels
(Ministère de la Santé et de la Protection sociale, OMS Maroc). Il ne pose pas de diagnostic
et ne remplace pas un professionnel de santé.

**Comment une réponse est produite**

1. La question est détectée en darija ou en français, puis traduite si besoin.
2. Les passages les plus proches sont recherchés (sémantique FAISS + mots-clés BM25, fusionnés par RRF).
3. Si aucun passage n'atteint le seuil de pertinence ({LOW_RELEVANCE_THRESHOLD}), l'assistant refuse de répondre au lieu d'inventer.
4. Sinon, le modèle rédige la réponse en citant ses sources, et un rappel médical est ajouté.

**Lire les scores de pertinence**
"""
    )
    st.markdown(
        '<span class="pill pill-teal">≥ 0,55 forte</span> '
        '<span class="pill">0,35 – 0,55 moyenne</span> '
        '<span class="pill pill-grey">&lt; 0,35 faible</span>',
        unsafe_allow_html=True,
    )
    st.markdown("&nbsp;")
    st.warning("En cas d'urgence, appelez le 141 (SAMU) ou le 15 (Protection civile). N'attendez pas une réponse de l'assistant.")


# ============================================================
# Point d'entrée
# ============================================================
inject_css()
init_state()
settings = render_sidebar()

view = st.session_state.view
if view == "Assistant":
    view_assistant(settings)
elif view == "Tableau de bord":
    view_dashboard()
else:
    view_about()
