import streamlit as st
import concurrent.futures
import time
from agent_logic import get_agentic_engine

# 1. Page Configuration
st.set_page_config(
    page_title="Agentic AI Research Lab",
    page_icon="🎓",
    layout="wide"
)

# 2. Professional Styling
st.markdown("""
    <style>
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .stStatusWidget { border: 1px solid #d1d1d1; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar - System Status
with st.sidebar:
    st.title("🎓 Project Dashboard")
    st.success("Database: ChromaDB (Active)")
    st.success("LLM: Llama 3.3-70B (Groq)")
    st.success("Context: 128k (Reranked)")
    
    st.divider()
    st.info("""
    **How it works:**
    1. Query is analyzed.
    2. Agent decomposes into sub-tasks.
    3. Hybrid Search finds snippets.
    4. 8B model reranks for accuracy.
    5. 70B model synthesizes final answer.
    """)
    
    if st.button("Clear Research History"):
        st.session_state.messages = []
        st.rerun()

# 4. Load the Agentic Engine
@st.cache_resource
def load_system():
    return get_agentic_engine()

try:
    agent = load_system()
except Exception as e:
    st.error(f"System Load Error: {e}")
    st.stop()

# 5. Threaded Query Runner (M2 Optimization)
def run_agent_query(user_input):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(agent.query, user_input)
        return future.result()

# 6. Main UI Logic
st.title("🔬 Agentic AI Research Assistant")
st.caption("Perform deep comparative analysis across your academic PDF library.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask a comparative research question..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Show the "Thinking" Trace
        with st.status("🚀 Agent is decomposing query and reranking nodes...", expanded=True) as status:
            try:
                # Execute the reasoning loop
                response = run_agent_query(prompt)
                
                final_text = str(response)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                # Render the final synthesis
                response_placeholder.markdown(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                
            except Exception as e:
                status.update(label="Rate Limit or Error", state="error")
                if "429" in str(e):
                    st.error("Groq Rate Limit Reached. Please wait 30 seconds.")
                else:
                    st.error(f"Error: {e}")