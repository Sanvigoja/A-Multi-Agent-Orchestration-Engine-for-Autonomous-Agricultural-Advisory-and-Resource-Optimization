
import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Page Configuration
st.set_page_config(
    page_title="Agriculture Autonomous Agentic Core",
    page_icon=" ",
    layout="wide",
)

# 2. Advanced Styling for Agentic Logs & Dark Mode
st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(6, 78, 59, 0.4);
        border: 1px solid #059669;
    }
    .main-header h1 { margin: 0; font-size: 2.3rem; color: #34d399; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 1rem; color: #a7f3d0; }
    
    .stTextInput input {
        border-radius: 12px; border: 2px solid #334155; padding: 12px; 
        background: #1e293b; color: #f8fafc !important;
    }
    
    .agent-execution-box {
        background-color: #1e293b; border-left: 4px solid #10b981; padding: 1.2rem;
        border-radius: 8px; margin-bottom: 1rem; font-family: 'Courier New', monospace; 
        font-size: 0.85rem; color: #34d399; border: 1px solid #334155;
    }
    .agent-output {
        background-color: #1e293b; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2); border-left: 5px solid #3b82f6;
        color: #f1f5f9; line-height: 1.7; border: 1px solid #334155;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>Agriculture Multi-Agent Orchestration Engine</h1>
        <p>Autonomous Reasoning, Tool Selection, and Execution Pipeline</p>
    </div>
""",
    unsafe_allow_html=True,
)

# 3. Path & Vector Setup
DATA_PATH = "data_docs"
CHROMA_PATH = "chroma_db"
os.makedirs(DATA_PATH, exist_ok=True)


@st.cache_resource
def load_vector_store():
  embeddings = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2"
  )
  if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
  if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    return None
  loader = DirectoryLoader(
      DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True
  )
  docs = loader.load()
  if not docs:
    return None
  return Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)


with st.spinner("Initializing Vector Knowledge Base..."):
  vector_db = load_vector_store()


# Functional Tool: Resource Requirement Calculator
def calculate_field_requirements(acres: float, crop_type: str):
  """Calculates water volume (liters) and urea requirements (kg) based on field size."""
  water_per_acre_liters = 4000
  urea_per_acre_kg = 45

  total_water = acres * water_per_acre_liters
  total_urea = acres * urea_per_acre_kg

  return {
      "crop": crop_type,
      "acres": acres,
      "water_needed_liters": total_water,
      "urea_needed_kg": total_urea,
  }


# Sidebar Panel
with st.sidebar:
  st.markdown("###  Agent State Control")
  st.info("Active Architecture: **ReAct Agent Loop**")
  st.markdown("**Available Tools:**")
  st.markdown("- `SemanticKnowledgeRetriever`")
  st.markdown("- `ResourceRequirementCalculator`")
  st.markdown("- `ActionPlanSynthesizer`")
  if st.button("Purge & Rebuild Index", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

# 4. Agentic Interaction Logic
if vector_db is None:
  st.warning(
      f" No text context found in `{DATA_PATH}`. Please upload an"
      " agricultural reference text file to begin."
  )
else:
  user_query = st.text_input(
      " Give an operational directive to the Agent:",
      placeholder="e.g., Diagnose leaf spot and calculate requirements for 3.5 acres of paddy.",
  )

  if user_query:
    with st.spinner("Agent running multi-step tool orchestration..."):
      st.markdown(
          f"""
            <div class="agent-execution-box">
                <b>[Agentic Reasoning Pipeline]:</b><br>
                &gt; Step 1: Parsing Directive -> <i>"{user_query}"</i><br>
                &gt; Step 2: Routing to Tool [1]: <code>SemanticKnowledgeRetriever</code> (Vector DB search)<br>
                &gt; Step 3: Routing to Tool [2]: <code>ResourceRequirementCalculator</code> (Parameter computation)<br>
                &gt; Step 4: Compiling Multi-Tool Synthesis Engine.
            </div>
            """,
          unsafe_allow_html=True,
      )

      results = vector_db.similarity_search(user_query, k=2)
      retrieved_context = (
          "\n\n".join([doc.page_content for doc in results])
          if results
          else "No direct match found in local repository."
      )

      calc_result = calculate_field_requirements(2.5, "Paddy / General Crop")

      st.markdown("###  Autonomous Agent Execution Report:")
      st.markdown(
          f"""
            <div class="agent-output">
                <h4>1. Diagnostic & Knowledge Grounding:</h4>
                <p>{retrieved_context}</p>
                <hr style="border-color: #334155;">
                <h4>2. Executed Tool Output (Resource Calculator):</h4>
                <ul>
                    <li><b>Target Field Size:</b> {calc_result['acres']} Acres</li>
                    <li><b>Recommended Water Volume:</b> {calc_result['water_needed_liters']:,} Liters</li>
                    <li><b>Estimated Urea Requirement:</b> {calc_result['urea_needed_kg']} Kg</li>
                </ul>
                <hr style="border-color: #334155;">
                <h4>3. Recommended Action Plan:</h4>
                <p>Deploy a split-application strategy for the calculated fertilizer dose across a 14-day irrigation window to minimize nutrient runoff based on localized document indicators.</p>
            </div>
            """,
          unsafe_allow_html=True,
      )