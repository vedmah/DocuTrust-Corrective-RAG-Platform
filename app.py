"""
app.py
------
DocuTrust: Enterprise Advanced RAG Platform with Automated Self-Correction.

A Streamlit front end for the CRAG (Corrective RAG) pipeline defined in
rag_engine.py. Split-pane layout:
  LEFT  -> document upload, live step-by-step agent evaluation log
  RIGHT -> validated, strictly-cited answers + conversation history

No API key required. Generation runs on a local LLM via Ollama
(https://ollama.com) -- fully free and offline.

Run with:
    ollama pull llama3.2      # one-time, downloads the local model
    ollama serve               # if not already running in the background
    streamlit run app.py
"""

import time

import streamlit as st

import rag_engine as rg


# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="DocuTrust | Corrective RAG Platform",
    page_icon="🛡️",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0f1116; }

    .docutrust-header {
        padding: 1.1rem 1.4rem;
        background: linear-gradient(135deg, #1a2740 0%, #101826 100%);
        border: 1px solid #2a3854;
        border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    .docutrust-header h1 {
        font-size: 1.5rem;
        margin: 0;
        color: #e8edf7;
        font-weight: 700;
    }
    .docutrust-header p {
        margin: 0.25rem 0 0 0;
        color: #8a9bbf;
        font-size: 0.88rem;
    }

    .pane-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7fa8;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid #232d42;
        padding-bottom: 0.4rem;
    }

    .trace-card {
        border-radius: 8px;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.45rem;
        border-left: 3px solid #364868;
        background: #161b27;
        font-size: 0.85rem;
    }
    .trace-card.status-running { border-left-color: #5b8def; background: #131c2e; }
    .trace-card.status-done    { border-left-color: #3ecf8e; }
    .trace-card.status-warning { border-left-color: #e8b339; }
    .trace-card.status-error   { border-left-color: #e85f5f; }

    .trace-agent {
        font-weight: 700;
        color: #cdd8ef;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .trace-message { color: #aab6d3; margin-top: 0.15rem; }
    .trace-detail {
        color: #6b7fa8;
        font-size: 0.76rem;
        margin-top: 0.25rem;
        font-style: italic;
    }

    .answer-box {
        background: #131826;
        border: 1px solid #232d42;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        color: #dde4f5;
        line-height: 1.55;
    }

    .citation-card {
        background: #161b27;
        border: 1px solid #232d42;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
    }
    .citation-marker {
        display: inline-block;
        background: #2a3854;
        color: #9fc3ff;
        border-radius: 4px;
        padding: 0.05rem 0.45rem;
        font-weight: 700;
        margin-right: 0.4rem;
        font-size: 0.75rem;
    }
    .citation-source { color: #cdd8ef; font-weight: 600; }
    .citation-score { color: #6b7fa8; font-size: 0.74rem; }
    .citation-preview { color: #8a9bbf; margin-top: 0.3rem; font-style: italic; }
    .citation-web-badge {
        background: #3a2e14; color: #e8b339; border-radius: 4px;
        padding: 0.05rem 0.4rem; font-size: 0.68rem; margin-left: 0.4rem;
    }

    .fallback-banner {
        background: #2a2410;
        border: 1px solid #4a3f17;
        color: #e8c95f;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stChatMessage"] { background: transparent; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resources (models loaded once per server process)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return rg.load_embedder()


@st.cache_resource(show_spinner="Loading cross-encoder grading model...")
def get_cross_encoder():
    return rg.load_cross_encoder()


def get_ollama_client():
    return rg.get_ollama_client()


# --------------------------------------------------------------------------- #
# Session state init
# --------------------------------------------------------------------------- #

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "indexed_filenames" not in st.session_state:
    st.session_state.indexed_filenames = []
if "trace_log" not in st.session_state:
    st.session_state.trace_log = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"query":..., "result": RAGResult}
if "processing" not in st.session_state:
    st.session_state.processing = False


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="docutrust-header">
        <h1>🛡️ DocuTrust</h1>
        <p>Self-correcting Retrieval-Augmented Generation for enterprise policy documents — cross-encoder graded retrieval, live agent trace, strict citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Layout: split pane
# --------------------------------------------------------------------------- #

left, right = st.columns([0.42, 0.58], gap="medium")


# ============================== LEFT PANE ================================= #
with left:
    st.markdown('<div class="pane-title">📂 Document Intake & Agent Trace</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Local model status (Ollama)", expanded=True):
        ollama_client = get_ollama_client()
        ollama_ok, ollama_msg = rg.check_ollama_available(ollama_client)
        if ollama_ok:
            st.success(f"✅ {ollama_msg} (model: `{rg.OLLAMA_MODEL}`)")
        else:
            st.error(ollama_msg)
            st.caption(
                "DocuTrust runs entirely free and offline using a local LLM via "
                "[Ollama](https://ollama.com) — no API key needed. "
                f"Install Ollama, then run `ollama pull {rg.OLLAMA_MODEL}` in a terminal."
            )

    uploaded_files = st.file_uploader(
        "Drop multi-page policy PDF(s) here",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        index_clicked = st.button("🔍 Build Vector Index", use_container_width=True, type="primary")
    with col_b:
        clear_clicked = st.button("🗑️ Clear Index", use_container_width=True)

    if clear_clicked:
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.indexed_filenames = []
        st.session_state.trace_log = []
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.indexed_filenames:
        st.caption(
            f"📚 Indexed: {', '.join(st.session_state.indexed_filenames)} "
            f"({len(st.session_state.chunks)} chunks)"
        )

    trace_placeholder = st.container(height=480)

    def render_trace_log():
        with trace_placeholder:
            trace_placeholder.empty()
            for ev in st.session_state.trace_log:
                icon = {"running": "⏳", "done": "✅", "warning": "⚠️", "error": "❌"}.get(ev.status, "•")
                detail_html = f'<div class="trace-detail">{ev.detail}</div>' if ev.detail else ""
                st.markdown(
                    f"""
                    <div class="trace-card status-{ev.status}">
                        <div class="trace-agent">{icon} {ev.agent}</div>
                        <div class="trace-message">{ev.message}</div>
                        {detail_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if index_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF before building the index.")
        else:
            st.session_state.trace_log = []
            embedder = get_embedder()

            def on_trace(ev: rg.TraceEvent):
                st.session_state.trace_log.append(ev)
                render_trace_log()

            docs = [(f.name, f.read()) for f in uploaded_files]
            with st.spinner("Running ingestion pipeline..."):
                index, chunks = rg.build_index(docs, embedder, on_trace)

            if index is not None:
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.indexed_filenames = [f.name for f in uploaded_files]
                st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")
                st.rerun()

    render_trace_log()


# ============================== RIGHT PANE ================================= #
with right:
    st.markdown('<div class="pane-title">💬 Validated, Cited Answers</div>', unsafe_allow_html=True)

    if st.session_state.index is None:
        st.info("Upload and index a document on the left to start asking questions.")
    else:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["query"])
            with st.chat_message("assistant"):
                result: rg.RAGResult = turn["result"]
                if result.used_web_fallback:
                    st.markdown(
                        '<div class="fallback-banner">⚠️ Local document context was insufficient — '
                        "the corrective agent rewrote the query and pulled supplementary context from a live web search.</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)

                if result.citations:
                    st.markdown("**Sources**")
                    for c in result.citations:
                        web_badge = '<span class="citation-web-badge">WEB</span>' if c["origin"] == "web" else ""
                        href_line = f'<br><a href="{c["href"]}" target="_blank">{c["href"]}</a>' if c.get("href") else ""
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <span class="citation-marker">{c['marker']}</span>
                                <span class="citation-source">{c['source']}</span>{web_badge}
                                <span class="citation-score"> &nbsp;relevance score: {c['score']}</span>
                                <div class="citation-preview">"{c['text_preview']}"{href_line}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        query = st.chat_input("Ask a question about the uploaded policy document(s)...")

        if query:
            ollama_client = get_ollama_client()
            ollama_ok, ollama_msg = rg.check_ollama_available(ollama_client)
            if not ollama_ok:
                st.error(ollama_msg)
            else:
                with st.chat_message("user"):
                    st.write(query)

                st.session_state.trace_log = []

                def on_trace(ev: rg.TraceEvent):
                    st.session_state.trace_log.append(ev)
                    render_trace_log()

                with st.chat_message("assistant"):
                    with st.spinner("Running Corrective RAG pipeline..."):
                        cross_encoder = get_cross_encoder()
                        embedder = get_embedder()
                        result = rg.run_crag_pipeline(
                            query=query,
                            index=st.session_state.index,
                            chunks=st.session_state.chunks,
                            embedder=embedder,
                            cross_encoder=cross_encoder,
                            ollama_client=ollama_client,
                            on_trace=on_trace,
                        )

                    if result.used_web_fallback:
                        st.markdown(
                            '<div class="fallback-banner">⚠️ Local document context was insufficient — '
                            "the corrective agent rewrote the query and pulled supplementary context from a live web search.</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)

                    if result.citations:
                        st.markdown("**Sources**")
                        for c in result.citations:
                            web_badge = '<span class="citation-web-badge">WEB</span>' if c["origin"] == "web" else ""
                            href_line = f'<br><a href="{c["href"]}" target="_blank">{c["href"]}</a>' if c.get("href") else ""
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <span class="citation-marker">{c['marker']}</span>
                                    <span class="citation-source">{c['source']}</span>{web_badge}
                                    <span class="citation-score"> &nbsp;relevance score: {c['score']}</span>
                                    <div class="citation-preview">"{c['text_preview']}"{href_line}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.chat_history.append({"query": query, "result": result})
                render_trace_log()
