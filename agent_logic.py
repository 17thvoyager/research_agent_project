import re
import time
import nest_asyncio
from llama_index.core import PromptTemplate
from llama_index.core import get_response_synthesizer
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.question_gen.output_parser import SubQuestionOutputParser
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from research_engine import get_research_engine
from app_config import get_dual_models


class CleanJSONOutputParser(SubQuestionOutputParser):
    """
    The Groq 70B model wraps JSON in ```json fences AND appends prose after
    the closing ```. The built-in parse_json_markdown strips the opening fence
    but chokes on trailing text. This override brace-counts to extract only
    the JSON object before passing it to the parent parser.
    """

    def parse(self, output: str):
        # 1. Strip opening ```json or ``` markers
        cleaned = re.sub(r"```(?:json)?\s*", "", output)

        # 2. Find the JSON object using brace-depth counting
        #    (handles nested objects robustly — no regex needed for the body)
        start = cleaned.find("{")
        if start == -1:
            return super().parse(cleaned)  # fallback: let original parser try

        depth, end = 0, start
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        json_only = cleaned[start : end + 1].strip()
        
        # Pydantic safeguard: if LLM returns null tool_name for conversational queries
        json_only = json_only.replace('"tool_name": null', '"tool_name": "research_database"')
        
        return super().parse(json_only)

# Explicitly define the context budget so PromptHelper never defaults to 4096.
# Math: 32768 (context window) - 2048 (output) - ~530 (prompt) = ~30,190 available ✅
PROMPT_HELPER = PromptHelper(
    context_window=32768,
    num_output=2048,
    chunk_overlap_ratio=0.1,
    chunk_size_limit=None,
)

# Apply nest_asyncio for Mac M2
try:
    nest_asyncio.apply()
except:
    pass

# --- CUSTOM SYNTHESIS PROMPT ---
CUSTOM_SYNTHESIS_PROMPT = PromptTemplate(
    "You are an expert AI Research Assistant. Answer ONLY from the provided context.\n\n"
    "RULE 0 — CONTEXT IS THE ONLY SOURCE (most important rule):\n"
    "  - If the context is empty, says 'Empty Response', or does not contain information\n"
    "    relevant to the question, respond ONLY with:\n"
    "    '❌ No relevant content found in the uploaded documents for this query.\n"
    "     Please make sure the correct PDF is uploaded and try again.'\n"
    "  - NEVER answer from your own training knowledge. NEVER guess or hallucinate.\n"
    "  - Every fact in your answer must be traceable to the context below.\n\n"
    "MANDATORY FORMATTING RULES — follow these exactly, no exceptions:\n\n"
    "RULE 1 — FIGURES & TABLES IN THE TEXT:\n"
    "  - The context is extracted from PDFs. Figures are NOT visible to you, only their captions or surrounding text.\n"
    "  - NEVER say 'refer to the figure', 'check the image', or 'see the table'. Instead, describe what the "
    "    caption and surrounding text tell you about the figure.\n"
    "  - If a table is described in the text, reproduce it as a clean Markdown table then write a 2-3 sentence "
    "    summary paragraph below it explaining what the data means.\n\n"
    "RULE 2 — COMPARISON QUESTIONS:\n"
    "  - If comparing papers or methodologies, the MAIN answer MUST be a Markdown table showing the comparison.\n"
    "  - After the table, write a short 2-3 sentence prose summary of the key takeaway.\n\n"
    "RULE 3 — RESEARCH GAPS (MANDATORY — never skip this section):\n"
    "  - At the end of EVERY response, include:\n\n"
    "### 🔍 Identified Research Gaps\n"
    "- **<Gap Title>:** <One sentence describing the unanswered question or limitation from the text.>\n"
    "- **<Gap Title>:** <Second gap.>\n"
    "- **<Gap Title>:** <Third gap or opportunity for future work.>\n\n"
    "  - This section is NOT optional. Base gaps only on what is missing or unclear in the provided context.\n\n"
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

    # Use 70B for sub-question decomposition.
    # CleanJSONOutputParser strips markdown fences the model adds around JSON.
    question_gen = LLMQuestionGenerator.from_defaults(
        llm=reasoning_llm,
        output_parser=CleanJSONOutputParser(),
    )

    # Build the synthesizer with an explicit PromptHelper so it NEVER
    # falls back to LlamaIndex's 4096-token default context window.
    synthesizer = get_response_synthesizer(
        llm=reasoning_llm,
        prompt_helper=PROMPT_HELPER,      # <-- hard-wires the 32768 context window
        text_qa_template=CUSTOM_SYNTHESIS_PROMPT,
        response_mode="compact",
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