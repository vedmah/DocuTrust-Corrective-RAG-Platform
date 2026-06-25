"""
rag_engine.py
--------------
Core logic for DocuTrust: a self-correcting Corrective RAG (CRAG) pipeline.

Pipeline stages (mirrors the CRAG pattern described in the project brief):
  1. INGEST    -> parse PDF(s), chunk text, embed with sentence-transformers, index in FAISS
  2. RETRIEVE  -> embed the user query, pull top-k candidate chunks from FAISS
  3. GRADE     -> score each retrieved chunk against the query with a local cross-encoder
  4. CORRECT   -> if grading says the local context is weak/insufficient, rewrite the query
                  and fall back to a live web search for supplementary context
  5. GENERATE  -> call Claude with ONLY the surviving (graded-relevant) chunks and require
                  strict inline citations back to source chunk IDs

Every stage emits structured "trace" events so the UI can render a real-time
step-by-step agent log, which is the centerpiece of the product brief.

No MongoDB / FastAPI server is used here -- this is a self-contained Streamlit
app, so client/session state and interaction traces live in Streamlit's
session_state instead of an external database. This keeps the project a
single deployable unit (matching your two-file project pattern) while
preserving the same conceptual architecture: documents + metadata, chunk
indices, and persisted trace logs.
"""

from __future__ import annotations

import io
import re
import time
import uuid
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback for older installs
    from duckduckgo_search import DDGS  # type: ignore

import anthropic


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"

CHUNK_SIZE_CHARS = 900          # ~ a few sentences of policy text per chunk
CHUNK_OVERLAP_CHARS = 150       # keeps clause continuity across chunk boundaries
TOP_K_RETRIEVE = 8              # candidates pulled from FAISS before grading
TOP_K_AFTER_GRADE = 5           # max graded chunks kept for generation
RELEVANCE_THRESHOLD = 0.0       # cross-encoder logit threshold; tuned for ms-marco-MiniLM
MIN_RELEVANT_CHUNKS = 2         # if fewer than this pass grading -> trigger web fallback
WEB_FALLBACK_RESULTS = 4


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance metadata."""
    chunk_id: str
    text: str
    source_doc: str
    page_number: int
    vector_idx: int = -1


@dataclass
class GradedChunk:
    chunk: Chunk
    score: float
    relevant: bool
    origin: str = "document"   # "document" or "web"


@dataclass
class TraceEvent:
    """One step of agent activity, rendered live in the UI log."""
    agent: str          # e.g. "Retriever", "Grader", "Query Rewriter", "Web Search", "Generator"
    status: str         # "running" | "done" | "warning" | "error"
    message: str
    detail: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RAGResult:
    answer: str
    citations: list[dict]
    graded_chunks: list[GradedChunk]
    used_web_fallback: bool
    trace: list[TraceEvent]


# --------------------------------------------------------------------------- #
# Model loading (cached by caller via st.cache_resource)
# --------------------------------------------------------------------------- #

def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def load_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


# --------------------------------------------------------------------------- #
# Ingestion: PDF -> chunks -> FAISS index
# --------------------------------------------------------------------------- #

def parse_pdf(file_bytes: bytes, filename: str) -> list[tuple[str, int]]:
    """Extract text per page from a PDF. Returns list of (text, page_number)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((text, i + 1))
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Simple sliding-window chunker over characters, snapped to sentence boundaries where possible."""
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # try to snap end to the nearest sentence boundary within a small window
        if end < n:
            window = text[end: end + 80]
            boundary = re.search(r"[.!?]\s", window)
            if boundary:
                end += boundary.end()
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [c for c in chunks if c]


def build_index(
    documents: list[tuple[str, bytes]],
    embedder: SentenceTransformer,
    on_trace: Callable[[TraceEvent], None],
) -> tuple[Optional[faiss.IndexFlatIP], list[Chunk]]:
    """
    documents: list of (filename, file_bytes)
    Returns a FAISS index (inner-product over normalized vectors = cosine sim) and the
    parallel list of Chunk metadata (vector i corresponds to chunks[i]).
    """
    on_trace(TraceEvent("Ingestion Agent", "running", f"Parsing {len(documents)} document(s)..."))

    all_chunks: list[Chunk] = []
    for filename, file_bytes in documents:
        try:
            pages = parse_pdf(file_bytes, filename)
        except Exception as e:
            on_trace(TraceEvent("Ingestion Agent", "error", f"Failed to parse {filename}", str(e)))
            continue

        for page_text, page_num in pages:
            for piece in chunk_text(page_text):
                all_chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4())[:8],
                        text=piece,
                        source_doc=filename,
                        page_number=page_num,
                    )
                )

    if not all_chunks:
        on_trace(TraceEvent("Ingestion Agent", "error", "No extractable text found in uploaded document(s)."))
        return None, []

    on_trace(TraceEvent(
        "Ingestion Agent", "running",
        f"Embedding {len(all_chunks)} structural chunks with {EMBED_MODEL_NAME.split('/')[-1]}..."
    ))

    texts = [c.text for c in all_chunks]
    vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    for i, c in enumerate(all_chunks):
        c.vector_idx = i

    on_trace(TraceEvent(
        "Ingestion Agent", "done",
        f"Indexed {len(all_chunks)} chunks from {len(documents)} document(s) into FAISS ({dim}-dim vectors)."
    ))

    return index, all_chunks


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    embedder: SentenceTransformer,
    on_trace: Callable[[TraceEvent], None],
    top_k: int = TOP_K_RETRIEVE,
) -> list[Chunk]:
    on_trace(TraceEvent("Retriever Agent", "running", f"Embedding query and searching FAISS index (top-{top_k})..."))

    qvec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qvec)

    k = min(top_k, len(chunks))
    scores, indices = index.search(qvec, k)

    retrieved = [chunks[i] for i in indices[0] if i >= 0]

    on_trace(TraceEvent(
        "Retriever Agent", "done",
        f"Retrieved {len(retrieved)} candidate chunk(s) from local vector store.",
        detail=", ".join(f"{c.source_doc} p.{c.page_number}" for c in retrieved[:5])
    ))
    return retrieved


# --------------------------------------------------------------------------- #
# Grading (Corrective RAG core step)
# --------------------------------------------------------------------------- #

def grade_chunks(
    query: str,
    candidates: list[Chunk],
    cross_encoder: CrossEncoder,
    on_trace: Callable[[TraceEvent], None],
    origin: str = "document",
) -> list[GradedChunk]:
    if not candidates:
        return []

    on_trace(TraceEvent(
        "Grading Agent", "running",
        f"Scoring {len(candidates)} chunk(s) for relevance with local cross-encoder..."
    ))

    pairs = [(query, c.text) for c in candidates]
    raw_scores = cross_encoder.predict(pairs)

    graded = []
    for c, score in zip(candidates, raw_scores):
        score_f = float(score)
        relevant = score_f > RELEVANCE_THRESHOLD
        graded.append(GradedChunk(chunk=c, score=score_f, relevant=relevant, origin=origin))

    graded.sort(key=lambda g: g.score, reverse=True)

    n_relevant = sum(1 for g in graded if g.relevant)
    status = "done" if n_relevant > 0 else "warning"
    on_trace(TraceEvent(
        "Grading Agent", status,
        f"{n_relevant}/{len(graded)} chunk(s) graded as relevant.",
        detail="; ".join(f"{g.chunk.source_doc} p.{g.chunk.page_number} -> score {g.score:.2f}" for g in graded[:5])
    ))

    return graded


# --------------------------------------------------------------------------- #
# Correction: query rewrite + web fallback
# --------------------------------------------------------------------------- #

def rewrite_query(query: str, claude_client: anthropic.Anthropic, on_trace: Callable[[TraceEvent], None]) -> str:
    on_trace(TraceEvent("Query Rewriter Agent", "running", "Local context insufficient. Rewriting query for web search..."))

    try:
        resp = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite the following question into a short, keyword-focused web search query "
                    "(no more than 10 words, no quotes, no preamble, just the query text):\n\n"
                    f"{query}"
                ),
            }],
        )
        rewritten = resp.content[0].text.strip().strip('"')
    except Exception as e:
        on_trace(TraceEvent("Query Rewriter Agent", "warning", "Rewrite failed, using original query.", str(e)))
        return query

    on_trace(TraceEvent("Query Rewriter Agent", "done", f"Rewritten search query: \"{rewritten}\""))
    return rewritten


def web_fallback_search(
    query: str,
    on_trace: Callable[[TraceEvent], None],
    max_results: int = WEB_FALLBACK_RESULTS,
) -> list[Chunk]:
    on_trace(TraceEvent("Web Search Agent", "running", f"Searching the web for supplementary context: \"{query}\"..."))

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        on_trace(TraceEvent("Web Search Agent", "error", "Web search failed.", str(e)))
        return []

    web_chunks = []
    for r in results:
        body = (r.get("body") or "").strip()
        title = r.get("title") or "Web result"
        href = r.get("href") or ""
        if not body:
            continue
        web_chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4())[:8],
                text=body,
                source_doc=f"web: {title}",
                page_number=0,
                vector_idx=-1,
            )
        )
        web_chunks[-1].href = href  # type: ignore[attr-defined]

    on_trace(TraceEvent(
        "Web Search Agent", "done" if web_chunks else "warning",
        f"Retrieved {len(web_chunks)} web result(s)."
    ))
    return web_chunks


# --------------------------------------------------------------------------- #
# Generation with strict citations
# --------------------------------------------------------------------------- #

def build_context_block(graded: list[GradedChunk]) -> str:
    lines = []
    for i, g in enumerate(graded, start=1):
        loc = f"{g.chunk.source_doc}, p.{g.chunk.page_number}" if g.origin == "document" else g.chunk.source_doc
        lines.append(f"[{i}] (source: {loc})\n{g.chunk.text}")
    return "\n\n".join(lines)


GENERATION_SYSTEM_PROMPT = """You are DocuTrust, an enterprise document Q&A assistant.
You must answer ONLY using the numbered context chunks provided. Rules:
1. Every factual sentence in your answer must end with a citation marker like [1], [2] referencing the chunk number(s) it came from.
2. If the provided context does not contain enough information to answer, say so explicitly. Do not guess or use outside knowledge.
3. Be concise and precise -- this is for corporate policy compliance, accuracy matters more than length.
4. Do not fabricate chunk numbers. Only cite chunks that were actually provided.
"""


def generate_answer(
    query: str,
    graded: list[GradedChunk],
    claude_client: anthropic.Anthropic,
    on_trace: Callable[[TraceEvent], None],
) -> tuple[str, list[dict]]:
    on_trace(TraceEvent("Generator Agent", "running", "Generating cited answer from validated context with Claude..."))

    context_block = build_context_block(graded)
    user_msg = f"Context chunks:\n\n{context_block}\n\nQuestion: {query}"

    try:
        resp = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=GENERATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        answer = resp.content[0].text.strip()
    except Exception as e:
        on_trace(TraceEvent("Generator Agent", "error", "Generation failed.", str(e)))
        return f"Generation failed: {e}", []

    citations = []
    for i, g in enumerate(graded, start=1):
        marker = f"[{i}]"
        if marker in answer:
            loc = f"{g.chunk.source_doc}, p.{g.chunk.page_number}" if g.origin == "document" else g.chunk.source_doc
            citations.append({
                "marker": marker,
                "source": loc,
                "origin": g.origin,
                "score": round(g.score, 3),
                "text_preview": g.chunk.text[:220] + ("..." if len(g.chunk.text) > 220 else ""),
                "href": getattr(g.chunk, "href", None),
            })

    on_trace(TraceEvent("Generator Agent", "done", f"Answer generated with {len(citations)} cited source(s)."))
    return answer, citations


# --------------------------------------------------------------------------- #
# Orchestration: the full CRAG pipeline for a single query
# --------------------------------------------------------------------------- #

def run_crag_pipeline(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    embedder: SentenceTransformer,
    cross_encoder: CrossEncoder,
    claude_client: anthropic.Anthropic,
    on_trace: Callable[[TraceEvent], None],
) -> RAGResult:
    """Runs retrieve -> grade -> (correct if needed) -> generate, emitting trace events throughout."""

    trace: list[TraceEvent] = []

    def emit(ev: TraceEvent):
        trace.append(ev)
        on_trace(ev)

    # 1. Retrieve
    candidates = retrieve(query, index, chunks, embedder, emit)

    # 2. Grade
    graded = grade_chunks(query, candidates, cross_encoder, emit, origin="document")
    relevant = [g for g in graded if g.relevant][:TOP_K_AFTER_GRADE]

    used_web_fallback = False

    # 3. Correct, if local grading is insufficient
    if len(relevant) < MIN_RELEVANT_CHUNKS:
        used_web_fallback = True
        emit(TraceEvent(
            "Corrective Controller", "warning",
            f"Only {len(relevant)} relevant chunk(s) found locally (need {MIN_RELEVANT_CHUNKS}). Triggering correction."
        ))

        rewritten = rewrite_query(query, claude_client, emit)
        web_chunks = web_fallback_search(rewritten, emit)

        if web_chunks:
            web_graded = grade_chunks(query, web_chunks, cross_encoder, emit, origin="web")
            relevant_web = [g for g in web_graded if g.relevant]
            relevant = (relevant + relevant_web)[:TOP_K_AFTER_GRADE]
            emit(TraceEvent(
                "Corrective Controller", "done",
                f"Merged {len(relevant_web)} web-sourced chunk(s) into context. Total context: {len(relevant)} chunk(s)."
            ))
        else:
            emit(TraceEvent("Corrective Controller", "warning", "Web fallback returned no usable results."))

    if not relevant:
        emit(TraceEvent("Corrective Controller", "error", "No relevant context found in document or web fallback."))
        return RAGResult(
            answer="I could not find sufficient relevant information in the uploaded document(s) or the web to answer this question confidently.",
            citations=[],
            graded_chunks=[],
            used_web_fallback=used_web_fallback,
            trace=trace,
        )

    # 4. Generate with strict citations
    answer, citations = generate_answer(query, relevant, claude_client, emit)

    return RAGResult(
        answer=answer,
        citations=citations,
        graded_chunks=relevant,
        used_web_fallback=used_web_fallback,
        trace=trace,
    )
