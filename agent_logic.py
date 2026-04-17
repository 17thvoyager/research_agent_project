import time
import nest_asyncio
from llama_index.core import PromptTemplate
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from research_engine import get_research_engine
from app_config import get_dual_models

# Apply nest_asyncio for Mac M2
try:
    nest_asyncio.apply()
except:
    pass

# --- PHASE B: THE CUSTOM SYNTHESIS PROMPT ---
CUSTOM_SYNTHESIS_PROMPT = PromptTemplate(
    "You are an expert AI Research Assistant. You have gathered evidence from multiple sub-queries.\n"
    "Using the provided context below, answer the user's main question.\n\n"
    "CRITICAL FORMATTING RULES:\n"
    "1. If the user asks to 'compare' papers or methodologies, your primary answer MUST be a Markdown Table. Do not write long paragraphs.\n"
    "2. At the very end of your response, add a section called '### 🔍 Identified Research Gaps'.\n"
    "3. In the Research Gaps section, list exactly two unanswered questions or limitations based ONLY on the provided text.\n\n"
    "Context Information:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "User Query: {query_str}\n"
    "Final Answer:\n"
)

class ThrottledAgent(SubQuestionQueryEngine):
    def query(self, str_or_query_bundle):
        time.sleep(2) # Protect Groq Quota
        return super().query(str_or_query_bundle)

def get_agentic_engine():
    reasoning_llm, fast_llm, embed_model = get_dual_models()
    base_engine = get_research_engine()

    tools = [
        QueryEngineTool(
            query_engine=base_engine,
            metadata=ToolMetadata(
                name="research_database",
                description="Search facts across research papers."
            ),
        ),
    ]

    # Use 70B for the "Thinking" phase (Sub-questions)
    question_gen = LLMQuestionGenerator.from_defaults(llm=reasoning_llm)

    # --- THE FIX: Explicitly build the Synthesizer with our Prompt ---
    # We use "compact" mode to save Groq tokens by packing context tightly
    synthesizer = get_response_synthesizer(
        llm=reasoning_llm, 
        text_qa_template=CUSTOM_SYNTHESIS_PROMPT,
        response_mode="compact"
    )

    # Create the Agent and inject the entire Synthesizer object
    agent_engine = ThrottledAgent.from_defaults(
        question_gen=question_gen,
        query_engine_tools=tools,
        response_synthesizer=synthesizer, # <-- INJECTED HERE
        use_async=False
    )

    return agent_engine

if __name__ == "__main__":
    agent = get_agentic_engine()
    print("\n🚀 AGENTIC AI (WITH RESEARCH GAPS) READY")
    
    # Test the new Table + Gaps feature
    query = "Compare the main algorithms or methodologies used across the papers."
    print(f"\nUser Query: {query}")
    print("Agent is thinking...\n")
    
    try:
        response = agent.query(query)
        print(f"\nFinal Response:\n{response}")
    except Exception as e:
        print(f"\n❌ Error during query: {e}")