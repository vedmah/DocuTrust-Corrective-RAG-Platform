import streamlit as st  # Fixed: Removed the accidental duplicate 'import streamlit as pd' typo
import time
import random

# 1. Page Configuration (Wide Layout for Split-Pane)
st.set_page_config(
    page_title="DocuTrust | Enterprise Advanced CRAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header
st.title("🛡️ DocuTrust")
st.caption("Enterprise Advanced RAG Platform with Automated Self-Correction & Verification")
st.markdown("---")

# 2. Initialize Session State (Fixes the disappearing output bug)
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None
if "meta" not in st.session_state:
    st.session_state.meta = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# 3. Simulated CRAG Backend Agents
def simulate_crag_pipeline(query, document_name):
    """Simulates the LangGraph / CrewAI Corrective RAG flow with real-time logging"""
    
    # Step 1: Retrieval Agent
    yield "status", "🔄 Agent 1: Retrieving structural text chunks...", None
    time.sleep(1.2)
    chunks_found = random.choice([True, False]) # Simulating whether local doc has the answer
    yield "log", f"✅ Retrieved 4 semantic chunks from `{document_name}`.", None

    # Step 2: Grading Agent (Cross-Encoder)
    yield "status", "🧠 Agent 2: Cross-checking document relevance via Cross-Encoder...", None
    time.sleep(1.5)
    
    if chunks_found:
        yield "log", "🔥 High relevance score detected (0.89). Proceeding to generation.", None
        confidence = "HIGH (Local Document)"
        source = f"{document_name}, Page 4, Section 2.1"
        answer = f"Based on the official corporate packet `{document_name}`, all requests must be submitted 14 business days prior. Strict compliance is mandatory under Section 2.1."
    else:
        # Step 3: Correction Agent (Query Rewriter & Web Fallback)
        yield "log", "⚠️ Low relevance score detected (0.32). Local chunks insufficient.", None
        yield "status", "🔧 Agent 3: Rewriting query for fallback web search...", None
        time.sleep(1.0)
        rewritten_query = f"Enterprise corporate policy regulations for {query}"
        yield "log", f"🌐 Searching verified fallback databases for: *'{rewritten_query}'*", None
        time.sleep(1.5)
        confidence = "EXTERNALLY VERIFIED (Web Fallback)"
        source = "Regulatory Compliance Portal (Internal Fallback Index)"
        answer = "Local documents lacked this specific update, but verified fallback tracking indicates that standard compliance processing times have been extended to 21 days for Q3/Q4."

    # Step 4: Generation Agent
    yield "status", "✍️ Agent 4: Synthesizing final response and citations...", None
    time.sleep(1.0)
    
    yield "result", answer, {"confidence": confidence, "source": source}

# 4. UI Layout: Split-Pane Design
col1, col2 = st.columns([1, 1], gap="large")

# LEFT PANE: Document Upload & Agent Evaluation Logs
with col1:
    st.subheader("📁 Document Ingestion & Trace Logs")
    uploaded_file = st.file_uploader("Drop multi-page corporate PDFs here", type=["pdf"])
    
    if uploaded_file:
        st.success(f"Successfully indexed: `{uploaded_file.name}`")
        
        # Metadata Viewer (Simulating MongoDB metadata tracking)
        with st.expander("📊 Document Metadata (MongoDB State)"):
            st.json({
                "document_id": "doc_98234_x",
                "filename": uploaded_file.name,
                "file_size_kb": round(uploaded_file.size / 1024, 2),
                "status": "Indexed",
                "vector_chunks": 42
            })
            
    st.markdown("### 🪵 Real-Time Agent Evaluation Logs")
    # Form an explicit placeholder container so logs don't jump around out of order
    log_container = st.empty()

# RIGHT PANE: Querying & Validated Outputs
with col2:
    st.subheader("💬 Verified Query Desk")
    user_query = st.text_input(
        "Ask a policy or compliance question:", 
        placeholder="e.g., What is the submission timeline for travel expense reports?",
        disabled=not uploaded_file
    )
    
    if not uploaded_file:
        st.info("💡 Please upload a corporate PDF on the left pane to unlock the query desk.")
        
    submit_button = st.button("Run Verified Search", type="primary", disabled=not uploaded_file)

# 5. Interaction Controller & Execution Loop
if submit_button and user_query:
    st.session_state.pipeline_running = True
    
    # Target the empty log container on the left column explicitly
    with log_container.status("🚀 Initializing CRAG Multi-Agent Pipeline...", expanded=True) as status:
        for output_type, payload, meta in simulate_crag_pipeline(user_query, uploaded_file.name):
            if output_type == "status":
                status.update(label=payload)
            elif output_type == "log":
                st.write(payload)
            elif output_type == "result":
                status.update(label="✅ Pipeline Execution Complete!", state="complete")
                # Save results persistently to session state
                st.session_state.final_answer = payload
                st.session_state.meta = meta
                
    st.session_state.pipeline_running = False

# 6. Persistent Output Rendering (Keeps the right pane visible during page reruns)
if st.session_state.final_answer and not st.session_state.pipeline_running:
    with col2:
        st.markdown("---")
        st.markdown("### 🤖 Validated Output")
        st.write(st.session_state.final_answer)
        
        # Strict Citations Section
        st.markdown("#### 📑 Strict Citations & Trust Metrics")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Source Confidence", value=st.session_state.meta["confidence"])
        with c2:
            st.info(f"**Verified Source:**\n{st.session_state.meta['source']}")
