# app.py
import streamlit as st
import pandas as pd
from graph import crag_app

# 1. Page Configuration Setup (Wide Premium Dashboard Mode)
st.set_page_config(
    page_title="DocuTrust | Advanced Enterprise CRAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Title Layout
st.title("🛡️ DocuTrust Enterprise Workspace")
st.caption("Production Level Multi-Agent Control Center with Self-Correction Analytics")
st.markdown("---")

# 2. Session State Management (Fixes the vanishing log and chart reset bugs)
if "logs_list" not in st.session_state:
    st.session_state.logs_list = []
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None
if "meta_metrics" not in st.session_state:
    st.session_state.meta_metrics = None
if "chart_data_latency" not in st.session_state:
    st.session_state.chart_data_latency = {}
if "chart_data_tokens" not in st.session_state:
    st.session_state.chart_data_tokens = {}

# 3. Layout Segregation: Tabs for Workspace vs. Deep Analytics
tab1, tab2 = st.tabs(["🚀 Live Agent Workspace", "📊 System Analytics Dashboard"])

with tab1:
    # Main Split-Pane Column Layout
    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT PANE: Document Upload and Persistent Execution Logging
    with col1:
        st.subheader("📁 Core Document Repository")
        uploaded_file = st.file_uploader("Upload policy packets (PDF)", type=["pdf"])
        
        if uploaded_file:
            st.success(f"Indexed Asset Hash: `{uploaded_file.name}`")
            
        st.markdown("### 🪵 Real-Time Agent Trace Logger")
        
        # Render the persistent logs inside a structured UI container box
        log_box = st.container(border=True)
        if st.session_state.logs_list:
            with log_box:
                for past_log in st.session_state.logs_list:
                    st.markdown(past_log)
        else:
            log_box.info("Awaiting execution trigger... Trace records will stream here.")

    # RIGHT PANE: User Query Portal & Strict Verification Output Frame
    with col2:
        st.subheader("💬 Verified Compliance Desk")
        user_query = st.text_input(
            "Enter compliance validation parameters:", 
            placeholder="e.g., policy constraints regarding travel expense report submissions...",
            disabled=not uploaded_file
        )
        
        submit_button = st.button("Query Agent Network", type="primary", disabled=not uploaded_file or not user_query)

        # Execution Processing Trigger Loop
        if submit_button and user_query:
            # Reset state for clean run execution updates
            st.session_state.logs_list = []
            st.session_state.chart_data_latency = {}
            st.session_state.chart_data_tokens = {}
            
            initial_graph_inputs = {"query": user_query, "documents": []}
            
            with log_box:
                with st.status("🛠 Running Graph Agents...", expanded=True) as status_indicator:
                    for graph_step in crag_app.stream(initial_graph_inputs):
                        for node_id, agent_out in graph_step.items():
                            agent_name = agent_out.get("current_agent", node_id.upper())
                            log_text = agent_out.get("log_message", f"Finished node: {node_id}")
                            
                            # Update active indicator status label text string
                            status_indicator.update(label=f"🔄 Processing: {agent_name}")
                            
                            # Append directly onto interface viewport frames and save to memory space
                            formatted_log = f"**{agent_name}**: {log_text}"
                            st.markdown(formatted_log)
                            st.session_state.logs_list.append(formatted_log)
                            
                            # Accumulate latency and token diagnostic metrics for real-time charting
                            node_metrics = agent_out.get("metrics", {})
                            for metric_name, value in node_metrics.items():
                                if "Latency" in metric_name:
                                    st.session_state.chart_data_latency[agent_name] = value
                                if "Tokens" in metric_name:
                                    st.session_state.chart_data_tokens[agent_name] = value
                                    
                            if "generation" in agent_out:
                                st.session_state.final_answer = agent_out["generation"]
                                st.session_state.meta_metrics = {
                                    "confidence": agent_out.get("confidence", "UNKNOWN"),
                                    "source": agent_out.get("source", "N/A")
                                }
                    status_indicator.update(label="✅ Processing Engine Settled", state="complete")
            st.rerun() # Forces page to update so metrics instantly bind out to dashboard tabs smoothly

        # Render Final Structural Response Blocks
        if st.session_state.final_answer:
            st.markdown("---")
            st.markdown("### 🤖 Validated Engine Output")
            st.info(st.session_state.final_answer)
            
            st.markdown("#### 📑 Verification Citations")
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric(label="Calculated Attestation Confidence", value=st.session_state.meta_metrics["confidence"])
            with mc2:
                st.warning(f"**Verified Source Path:**\n{st.session_state.meta_metrics['source']}")

# 4. TAB 2: SYSTEM ANALYTICS DASHBOARD
with tab2:
    st.subheader("📊 Multi-Agent Compute & Audit Performance")
    
    if st.session_state.chart_data_latency:
        # High-level summary scorecard layout blocks
        total_latency = sum(st.session_state.chart_data_latency.values())
        total_tokens = sum(st.session_state.chart_data_tokens.values())
        
        card1, card2, card3 = st.columns(3)
        card1.metric("Total Processing Run Time", f"{total_latency:.2f} seconds")
        card2.metric("Total Context Token Window", f"{total_tokens:,} tokens")
        card3.metric("Agent Nodes Invoked", f"{len(st.session_state.chart_data_latency)} agents")
        
        st.markdown("---")
        
        # Split row grid framework mapping out bar and line analytics layout panels
        graph_col1, graph_col2 = st.columns(2)
        
        with graph_col1:
            st.markdown("##### ⏱ Operational Latency Breakdown (Per Agent)")
            # Standardize payload parsing out directly to pandas layout series tables
            latency_df = pd.DataFrame(
                list(st.session_state.chart_data_latency.items()), 
                columns=["Agent Node", "Execution Time (Seconds)"]
            ).set_index("Agent Node")
            st.bar_chart(latency_df, color="#2E86C1")
            
        with graph_col2:
            st.markdown("##### 🔤 Computational Token Processing Load")
            token_df = pd.DataFrame(
                list(st.session_state.chart_data_tokens.items()), 
                columns=["Agent Node", "Token Count"]
            ).set_index("Agent Node")
            st.area_chart(token_df, color="#2ECC71")
            
    else:
        st.info("📊 Run a verification query within the workspace portal to populate interactive chart metrics.")
