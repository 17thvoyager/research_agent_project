import os
import chromadb
from collections import Counter
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from app_config import get_dual_models

def generate_quality_report(documents):
    """
    Data Engineering step: Audits the extracted text BEFORE saving it to the database.
    """
    print("\n" + "="*55)
    print("📊 DATA INGESTION QUALITY REPORT")
    print("="*55)
    
    if not documents:
        print("❌ CRITICAL WARNING: Zero documents loaded. Check your 'data/' folder.")
        print("="*55 + "\n")
        return

    file_counts = Counter()
    empty_files = set()
    low_text_files = set()

    # Analyze every chunk of text extracted
    for doc in documents:
        file_name = doc.metadata.get("file_name", "Unknown PDF")
        file_counts[file_name] += 1
        
        # Check text density
        word_count = len(doc.text.split())
        if word_count == 0:
            empty_files.add(file_name)
        elif word_count < 10:  # Less than 10 words is suspiciously low
            low_text_files.add(file_name)

    print(f"Total Pages/Chunks Processed: {len(documents)}")
    print(f"Total Unique Files Found:   {len(file_counts)}\n")

    print("📄 File Breakdown:")
    for file, count in file_counts.items():
        print(f"  - {file}: {count} chunks")

    print("\n⚠️ Quality Warnings:")
    warnings_found = False
    
    if empty_files:
        print(f"  ❌ EMPTY FILES (No text extracted - check for DRM/Scans):")
        for f in empty_files: print(f"     -> {f}")
        warnings_found = True
        
    if low_text_files:
        print(f"  ⚠️ LOW TEXT FILES (Suspiciously low word count):")
        for f in low_text_files: print(f"     -> {f}")
        warnings_found = True
        
    if not warnings_found:
        print("  ✅ All files parsed successfully with healthy text density.")
    
    print("="*55 + "\n")

def run_advanced_ingestion():
    llm, fast_llm, embed_model = get_dual_models()
    print("\n⚡ Starting Data Pipeline...")

    # 1. Read PDFs using the fast PyMuPDF reader
    fast_reader = PyMuPDFReader()
    file_extractor = {".pdf": fast_reader}
    reader = SimpleDirectoryReader(input_dir="./data", file_extractor=file_extractor)
    
    print("📖 Ripping text from PDFs...")
    documents = reader.load_data()

    # 2. RUN THE QUALITY REPORT
    generate_quality_report(documents)

    # 3. Setup ChromaDB Persistence
    print("🗄️ Connecting to local ChromaDB...")
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("research_papers")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Create Index (Vectorizing)
    print("🧠 Vectorizing data (Running locally on Mac M2)...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print(f"\n✅ SUCCESS: Vector database updated and saved!")

if __name__ == "__main__":
    run_advanced_ingestion()