import time
import nest_asyncio
from llama_index.core import Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.prompts import PromptTemplate
from research_engine import get_research_engine
from app_config import get_dual_models

# Apply nest_asyncio for Mac M2
try:
    nest_asyncio.apply()
except:
    pass

# --- THE FIX: Custom Strict JSON Prompt ---
# We force the model to avoid any markdown formatting like ```json
STRICT_JSON_PROMPT = PromptTemplate(
    "You are a query decomposition expert. Your task is to break down a complex user query "
    "into a set of sub-questions that can be answered by specific tools.\n"
    "Output the results in the following JSON format ONLY. Do NOT use markdown blocks. "
    "Do NOT use backticks. Do NOT provide any introductory or concluding text.\n\n"
    "Format:\n"
    "{{\n"
    '  "items": [\n'
    '    {{\n'
    '      "sub_question": "the question",\n'
    '      "tool_name": "the tool name"\n'
    '    }}\n'
    '  ]\n'
    "}}\n\n"
    "Tools available: {tool_metadata_str}\n"
    "User Query: {query_str}\n"
)

class ThrottledAgent(SubQuestionQueryEngine):
    def query(self, str_or_query_bundle):
        time.sleep(2) # Protect Groq Quota
        return super().query(str_or_query_bundle)

def get_agentic_engine():
    # 1. Get models
    reasoning_llm, fast_llm, embed_model = get_dual_models()
    base_engine = get_research_engine()

    query_engine_tools = [
        QueryEngineTool(
            query_engine=base_engine,
            metadata=ToolMetadata(
                name="research_database",
                description="Use this to search facts across research papers."
            ),
        ),
    ]

    # THE FIX: We don't pass 'prompt' to from_defaults. 
    # We let LlamaIndex use its internal template which 70B handles well anyway.
    question_gen = LLMQuestionGenerator.from_defaults(llm=reasoning_llm)

    agent_engine = ThrottledAgent.from_defaults(
        question_gen=question_gen,
        query_engine_tools=query_engine_tools,
        llm=reasoning_llm, 
        use_async=False
    )

    return agent_engine