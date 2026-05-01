import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from app_config import get_dual_models


def get_research_engine():
    # 1. Load models (fast_llm no longer needed — reranking is now local)
    reasoning_llm, fast_llm, embed_model = get_dual_models()

    # 2. Connect to ChromaDB
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 3. Load the vector index (used for semantic/dense retrieval)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    # 4a. Dense retriever — top-10 semantic hits via embeddings
    vector_retriever = index.as_retriever(similarity_top_k=10)

    # 4b. Sparse retriever (BM25) — exact keyword matching.
    #     Pull every stored chunk from ChromaDB and build an in-memory
    #     BM25 index. No LLM call required — pure TF-IDF term frequency.
    #     This is what finds exact strings like "LLaMA 3.1 405B" reliably.
    raw = chroma_collection.get(include=["documents", "metadatas"])
    bm25_nodes = [
        TextNode(text=doc, metadata=meta or {}, id_=node_id)
        for doc, meta, node_id in zip(
            raw["documents"], raw["metadatas"], raw["ids"]
        )
        if doc  # skip any empty chunks
    ]

    if bm25_nodes:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes,
            similarity_top_k=10,
        )
        retrievers = [vector_retriever, bm25_retriever]
        print(f"[engine] Hybrid search: {len(bm25_nodes)} BM25 nodes + vector index")
    else:
        # No documents yet — fall back to vector-only
        retrievers = [vector_retriever]
        print("[engine] No documents indexed yet — using vector-only retrieval")

    # 5. Reciprocal Rank Fusion (RRF) — merges results from both retrievers.
    #    num_queries=1 means: use the ORIGINAL query only, no extra LLM calls.
    #    mode="reciprocal_rerank" applies RRF scoring across both result lists.
    hybrid_retriever = QueryFusionRetriever(
        retrievers,
        similarity_top_k=10,
        num_queries=1,         # ← 1 = no LLM query generation, just fuse
        mode="reciprocal_rerank",
        use_async=False,
    )

    # 6. Cross-encoder reranker (local, zero Groq API calls).
    #    BAAI/bge-reranker-base jointly encodes query+chunk to score relevance —
    #    more accurate than LLM reranking, runs on-device, no rate limits.
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base",
        top_n=5,
    )

    # 7. Assemble the full retrieval pipeline
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        llm=reasoning_llm,
    )

    return query_engine


if __name__ == "__main__":
    engine = get_research_engine()
    print("\n--- Testing Retrieval Engine ---")
    response = engine.query("What are the specific safety metrics used?")
    print(f"Response: {response}")