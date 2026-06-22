# app.py
import streamlit as st
from graph import crag_app  # Import our compiled state machine engine layout directly

# 1. Page Configuration Setup (Optimized for Split-Pane Presentation)
st.set_page_config(
    page_title="DocuTrust | Enterprise Advanced CRAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Application Branding & Design Accents
st.title("🛡️ DocuTrust")
st.caption("Enterprise Advanced RAG Platform with Automated Self-Correction & Verification")
st.markdown("---")

# 2. Session State Array Management (Mitigates Streamlit Frame-Refresh State Loss)
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None
if "meta_metrics" not in st.session_state:
    st.session_state.meta_metrics = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# 3. Main Split-Pane Column Grid Infrastructure Definition
col1, col2 = st.columns([1, 1], gap="large")

# LEFT WORKSPACE PANE: Data Intake & Multi-Agent Network Logs
with col1:
    st.subheader("📁 Document Ingestion & Trace Logs")
    uploaded_file = st.file_uploader(
        "Drop multi-page corporate PDFs here to index context window records", 
        type=["pdf"]
    )
    
    if uploaded_file:
        st.success(f"Successfully indexed: `{uploaded_file.name}`")
        
        # Simulating active dynamic state display pulled straight from MongoDB tracking records
        with st.expander("📊 Document Metadata (MongoDB State)"):
            st.json({
                "document_id": "doc_98234_x",
                "filename": uploaded_file.name,
                "file_size_kb": round(uploaded_file.size / 1024, 2),
                "status": "Indexed & Vectorized",
                "vector_chunks": 42,
                "embedding_model": "bge-large-en-v1.5"
            })
            
    st.markdown("### 🪵 Real-Time Agent Evaluation Logs")
    # Setting an immutable data placeholder boundary container target to prevent viewport shifting
    log_container = st.empty()

# RIGHT WORKSPACE PANE: Secure Interaction Desk & Citations Delivery
with col2:
    st.subheader("💬 Verified Query Desk")
    user_query = st.text_input(
        "Ask a policy or compliance question:", 
        placeholder="e.g., What is the submission timeline for travel expense reports?",
        disabled=not uploaded_file
    )
    
    if not uploaded_file:
        st.info("💡 Please upload a corporate PDF on the left pane to unlock the query desk.")
        
    submit_button = st.button(
        "Run Verified Search", 
        type="primary", 
        disabled=not uploaded_file or not user_query
    )

# 4. Multi-Agent Engine Trigger Controller Loop
if submit_button and user_query and uploaded_file:
    st.session_state.pipeline_running = True
    
    # Initialize state dictionary variables to pass through graph entry point
    initial_graph_inputs = {"query": user_query, "documents": []}
    
    # Target the empty log container on the left column using Streamlit status widgets
    with log_container.status("🚀 Initializing CRAG Multi-Agent Pipeline...", expanded=True) as status_box:
        
        # Actively consume live stream step dictionaries emitted by our compiled LangGraph application
        for graph_step in crag_app.stream(initial_graph_inputs):
            for node_identifier, agent_response in graph_step.items():
                
                # Extract localized system metadata trackers populated inside graph nodes
                agent_title = agent_response.get("current_agent", node_identifier.upper())
                runtime_log = agent_response.get("log_message", f"Completed loop step: {node_identifier}")
                
                # Stream logs line-by-line inside the status dropdown box
                status_box.update(label=f"🔄 Active Node: {agent_title}...")
                st.write(runtime_log)
                
                # Intercept generation structures to populate UI output frames
                if "generation" in agent_response:
                    st.session_state.final_answer = agent_response["generation"]
                    st.session_state.meta_metrics = {
                        "confidence": agent_response.get("confidence", "UNKNOWN"),
                        "source": agent_response.get("source", "N/A")
                    }
                    
        status_box.update(label="✅ Pipeline Execution Complete!", state="complete")
        
    st.session_state.pipeline_running = False

# 5. Persistent View Rendering Engine (Maintains outputs on browser re-renders)
if st.session_state.final_answer and not st.session_state.pipeline_running:
    with col2:
        st.markdown("---")
        st.markdown("### 🤖 Validated Output")
        st.info(st.session_state.final_answer)
        
        # Strict Citations & System Reliability Metric Dashboards
        st.markdown("#### 📑 Strict Citations & Trust Metrics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Source Confidence", value=st.session_state.meta_metrics["confidence"])
        with metric_col2:
            st.warning(f"**Verified Source Trace:**\n{st.session_state.meta_metrics['source']}")
