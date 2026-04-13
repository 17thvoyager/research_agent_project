import os
import chromadb
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from app_config import get_dual_models

def run_advanced_ingestion():
    # 1. Initialize models
    reasoning_llm, fast_llm, embed_model = get_dual_models()
    
    print("Starting Advanced Ingestion (Hybrid Ready)...")

    # 2. Configure LlamaParse (for Tables and Math)
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True
    )

    # 3. Read PDFs from the data folder
    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader(input_dir="./data", file_extractor=file_extractor)
    documents = reader.load_data()

    # 4. Setup ChromaDB (Vector Store)
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 5. Create the Vector Index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # We pass the embed_model explicitly to avoid OpenAI errors
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print(f"✅ Success: {len(documents)} nodes indexed in ChromaDB.")
    return index

if __name__ == "__main__":
    run_advanced_ingestion()