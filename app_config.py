import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

def get_dual_models():
    """Returns the 70B model for heavy reasoning and 8B for fast tasks.

    Groq rate limits (Llama 3.3-70B Versatile):
      - 6,000 tokens per minute
      - 100 requests per day
      - context window: 32,768 tokens (Groq free tier cap)
    We deliberately limit output to 2048 to leave room for context chunks.

    LlamaIndex PromptHelper math:
      available_context = context_window - prompt_tokens - num_output
      e.g. 32768 - 529 - 2048 = 30,191  ✅  (was 4096-529-4096 = -529 ❌)
    """
    api_key = os.getenv("GROQ_API_KEY")

    reasoning_llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,        # output cap; reduced from 4096 to leave room for context
        context_window=32768,   # tell LlamaIndex the real context window (Groq free tier)
    )
    fast_llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,        # reranker only needs short output
        context_window=32768,
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = reasoning_llm
    Settings.embed_model = embed_model
    Settings.context_window = 32768  # must match the LLM's real context window
    Settings.num_output = 2048       # max tokens the LLM will generate (matches max_tokens above)
    Settings.chunk_size = 256        # small chunks keep reranker calls within token budget
    Settings.chunk_overlap = 25

    return reasoning_llm, fast_llm, embed_model

def initialize_settings():
    get_dual_models()
    print("Success: Groq Llama Stack Configured!")