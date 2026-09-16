import os
import random
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CORPUS_DIR = "corpus"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# אתחול מודל ה-Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

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

def create_and_save_index(docs, chunk_size, chunk_overlap, save_dir):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_dir)
    print(f"🎉 Index saved to '{save_dir}/' with {len(chunks)} chunks (size={chunk_size}).")

if __name__ == "__main__":
    docs = load_documents(CORPUS_DIR)
    
    if docs:
        # 1. יצירת אינדקס קטן (Small - 300) לעובדות נקודתיות
        create_and_save_index(docs, chunk_size=300, chunk_overlap=50, save_dir="faiss_index_small")
        
        # 2. יצירת אינדקס גדול (Large - 1200) להקשר רחב ולשאילתות מורכבות
        create_and_save_index(docs, chunk_size=1200, chunk_overlap=200, save_dir="faiss_index_large")