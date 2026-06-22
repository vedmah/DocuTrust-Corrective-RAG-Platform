# graph.py
import os
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# AI/ML & Database Core Libraries
from sentence_transformers import CrossEncoder
import pymongo
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Define the Shared State Schema
class GraphState(TypedDict):
    query: str
    documents: List[str]
    current_agent: str
    log_message: str
    run_web_search: bool
    generation: str
    confidence: str
    source: str

# 2. Database & AI Models Initialization Setup
# MongoDB Trace Logging Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    mongo_client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = mongo_client["docutrust_db"]
    logs_collection = db["interaction_logs"]
except Exception:
    logs_collection = None  # Graceful fallback if Mongo isn't running locally

# Load Local Cross-Encoder Model for the Grading Agent
# This model scores document chunk relevance (0 to 1)
try:
    grading_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    grading_model = None

# Mock Database documents for backup lookup simulation
MOCK_WEB_KNOWLEDGE = {
    "travel policy": "Latest Q3 corporate policy update: International travel requires VP clearance 21 days out.",
    "expense submission": "All standard corporate travel and entertainment expenses must be logged within 30 days."
}

# 3. Agent Nodes Definition
def retrieve_chunks(state: GraphState) -> Dict[str, Any]:
    """Retrieves text components based on user question semantic queries."""
    query = state["query"]
    
    # In practice, you would run: vector_db.similarity_search(query, k=3)
    # Simulating standard structural chunk extraction for the blueprint:
    retrieved_chunks = [
        "Standard expense submissions must be filed through the corporate ledger system.",
        "Employees traveling on business must follow standard safety guidelines outlined in Packet A."
    ]
    
    return {
        "documents": retrieved_chunks,
        "current_agent": "Retrieval Agent",
        "log_message": f"📁 Step 1: Extracted {len(retrieved_chunks)} semantic data chunks from localized vector repository."
    }

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Evaluates chunk relevance using a localized cross-encoder rating."""
    query = state["query"]
    docs = state["documents"]
    
    if not grading_model or not docs:
        # Fallback if model hasn't downloaded completely yet
        return {"run_web_search": True, "current_agent": "Grading Agent", "log_message": "🧠 Step 2: Model warming up. Defaulting to verification fallback."}
    
    # Pair query with each document chunk for cross-encoder processing
    pairs = [[query, doc] for doc in docs]
    scores = grading_model.predict(pairs)
    max_score = max(scores) if len(scores) > 0 else 0
    
    # CRAG Decision Boundary: If highest chunk relevance score is lower than 0.0, trigger fallback search
    RELEVANCE_THRESHOLD = 0.0
    run_web = max_score < RELEVANCE_THRESHOLD
    
    log_msg = f"🧠 Step 2: Cross-Encoder validated document chunks. Max Relevance Score: {max_score:.2f}."
    if run_web:
        log_msg += " ⚠️ Below threshold! Routing to alternative fallback engine."
        
    return {
        "run_web_search": run_web,
        "current_agent": "Grading Agent",
        "log_message": log_msg
    }

def transform_query(state: GraphState) -> Dict[str, Any]:
    """Optimizes user inputs into specialized keywords for backup queries."""
    query = state["query"]
    optimized_keyword = query.lower().strip()
    
    return {
        "current_agent": "Query Rewriter Agent",
        "log_message": f"🔧 Step 3: Reformulated structural search terms for global web parsing: *'{optimized_keyword}'*"
    }

def web_search_fallback(state: GraphState) -> Dict[str, Any]:
    """Queries external endpoints or internal secondary indices if verification fails."""
    query = state["query"].lower()
    found_fallback = "No additional data records located across corporate fallback index structures."
    
    # Scan mock reference index to simulate alternative live data pulls
    for key, value in MOCK_WEB_KNOWLEDGE.items():
        if key in query:
            found_fallback = value
            break
            
    updated_docs = state["documents"] + [found_fallback]
    return {
        "documents": updated_docs,
        "current_agent": "Correction Agent (Web Fallback)",
        "log_message": "🌐 Step 4: Fallback search executed successfully. Supplemented context window with verified backup streams."
    }

def generate_answer(state: GraphState) -> Dict[str, Any]:
    """Synthesizes context arrays into a unified answer containing strict source indexes."""
    query = state["query"]
    docs = state["documents"]
    web_was_used = state.get("run_web_search", False)
    
    # In production, parse combined texts to an LLM context window here
    # Synthesizing structured response artifact:
    if web_was_used:
        generation = f"Synthesized via External Verification Index:\n\nRegarding '{query}', local documents lacked up-to-date entries, but alternative index arrays clarify that: {docs[-1]}"
        confidence = "CRITICAL VERIFIED (Fallback Loop)"
        source = "Regulatory Compliance Portal Index (Updated Q3)"
    else:
        generation = f"Synthesized from Secure Internal Documents:\n\nRegarding '{query}', localized policy sheets indicate processing follows regular intervals. All logs must clear standard filing frameworks."
        confidence = "SECURE LOCAL MATCH (High Trust)"
        source = "Internal Corporate Policy Packet, Sec 4.2"

    # Dynamic trace logging directly to MongoDB
    if logs_collection is not None:
        try:
            logs_collection.insert_one({
                "query": query,
                "confidence_score": confidence,
                "fallback_deployed": web_was_used,
                "source_attribution": source
            })
        except Exception:
            pass # Keep pipeline operational if DB connectivity drops

    return {
        "generation": generation,
        "confidence": confidence,
        "source": source,
        "current_agent": "Generation Agent",
        "log_message": "✍️ Step 5: Content synthesized into compliance format. Response delivery ready."
    }

# 4. Define and Link the Graph Edges Routing Logic
def decide_route(state: GraphState) -> str:
    if state["run_web_search"]:
        return "transform_query"
    return "generate_answer"

# 5. Build and Compile the Architectural Framework Setup
workflow = StateGraph(GraphState)

# Append Functional Processing Units (Nodes)
workflow.add_node("retrieve", retrieve_chunks)
workflow.add_node("grade", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search_fallback)
workflow.add_node("generate_answer", generate_answer)

# Connect Node Pipelines Electronically via Topological Directed Mapping
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")

# Branch Routing Decision Evaluation Injection
workflow.add_conditional_edges(
    "grade",
    decide_route,
    {
        "transform_query": "transform_query",
        "generate_answer": "generate_answer"
    }
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("generate_answer", END)

# Export compiled production application context execution engine hook
crag_app = workflow.compile()
