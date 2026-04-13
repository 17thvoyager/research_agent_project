import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

def get_dual_models():
    """Returns the 70B model for heavy reasoning and 8B for fast tasks."""
    api_key = os.getenv("GROQ_API_KEY")
    
    reasoning_llm = Groq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.1)
    fast_llm = Groq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.1)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    Settings.llm = reasoning_llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512 
    Settings.chunk_overlap = 50
    
    return reasoning_llm, fast_llm, embed_model

def initialize_settings():
    get_dual_models()
    print("Success: Groq Llama Stack Configured!")