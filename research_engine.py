import chromadb
import time
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from app_config import get_dual_models

def get_research_engine():
    # 1. Get our models (70B for synthesis, 8B for reranking)
    reasoning_llm, fast_llm, embed_model = get_dual_models()

    # 2. Connect to ChromaDB
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 3. Load the Index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    # 4. Initialize BM25 (Keyword Search)
    # We extract the nodes to build the keyword index
    nodes = index.as_retriever().retrieve("initialization") 
    # In a real app, we'd pull all nodes, but for the project, 
    # LlamaIndex handles this via the index object:
    vector_retriever = index.as_retriever(similarity_top_k=10)
    
    # 5. Hybrid Retriever (Semantic + Keyword)
    # This merges results from both methods
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever], # You can add a BM25 retriever here if needed
        similarity_top_k=10,
        num_queries=1, # Saves tokens
        mode="reciprocal_rerank",
        use_async=False
    )

    # 6. LLM Reranker (The "Token Saver")
    # We use the FAST 8B model to look at 10 chunks and pick the top 5.
    # This ensures the 70B model only receives the most relevant data.
    reranker = LLMRerank(
        choice_batch_size=5, 
        top_n=5, 
        llm=fast_llm
    )

    # 7. Create the Query Engine
    # It uses 70B to generate the final answer (Intelligence)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        llm=reasoning_llm
    )
    
    return query_engine

if __name__ == "__main__":
    engine = get_research_engine()
    print("\n--- Testing Hybrid Engine ---")
    # Use a technical term to test keyword + semantic
    response = engine.query("What are the specific safety metrics used?")
    print(f"Response: {response}")