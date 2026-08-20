import os
import random
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CORPUS_DIR = "corpus"
INDEX_DIR = "index"

def load_documents(corpus_dir):
    documents = []
    if not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)
        print(f"⚠️ Created empty '{corpus_dir}' directory. Please add your PDF/MD files there!")
        return documents

    for file_name in os.listdir(corpus_dir):
        file_path = os.path.join(corpus_dir, file_name)
        if file_name.endswith(".pdf"):
            print(f"📄 Loading PDF: {file_name}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Enrich metadata
            for doc in docs:
                doc.metadata["doc_name"] = file_name
                doc.metadata["page"] = str(doc.metadata.get("page", 0) + 1)
            documents.extend(docs)
        elif file_name.endswith(".md") or file_name.endswith(".txt"):
            print(f"📝 Loading Text/MD: {file_name}")
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["doc_name"] = file_name
                doc.metadata["page"] = "N/A"
            documents.extend(docs)
    
    print(f"✅ Loaded {len(documents)} document pages/files in total.")
    return documents

def build_and_save_index():
    # 1. Parse
    docs = load_documents(CORPUS_DIR)
    if not docs:
        print("❌ No documents found. Add files to 'corpus/' and re-run.")
        return

    # 2. Chunk (Baseline settings)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} text chunks.")

    # 3. Embed
    print("⏳ Initializing Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # 4. Store
    print("💾 Indexing chunks into FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"🎉 Index successfully saved to '{INDEX_DIR}/'!")

    # Task 3 Requirement: Inspection & Debugging
    print("\n" + "="*50)
    print("🔍 TASK 3 INSPECTION: 3 Random Chunks")
    print("="*50)
    sample_chunks = random.sample(chunks, min(3, len(chunks)))
    for i, chunk in enumerate(sample_chunks, 1):
        doc_name = chunk.metadata.get("doc_name", "Unknown")
        page = chunk.metadata.get("page", "N/A")
        print(f"\n--- Chunk Sample #{i} [{doc_name} | Page {page}] ---")
        print(chunk.page_content[:300] + "...\n")

if __name__ == "__main__":
    build_and_save_index()