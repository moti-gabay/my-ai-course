import os
import time
from anthropic import Anthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# 1. הגדרות נתיבים ומודלים
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# 2. טעינת מודל ה-Embeddings ושני אינדקסי ה-FAISS (Small + Large)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vs_small = FAISS.load_local(
    "faiss_index_small", 
    embeddings, 
    allow_dangerous_deserialization=True
)
vs_large = FAISS.load_local(
    "faiss_index_large", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. אתחול הלקוחות והמודלים
anthropic_client = Anthropic()
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_multi_scale_with_rerank(question: str, top_k_per_index: int = 10, final_top_k: int = 5) -> list:
    """
    1. שולף top_k מאינדקס Small ו-top_k מאינדקס Large.
    2. מאחד את המסמכים ומסיר כפילויות.
    3. מדרג מחדש (Rerank) עם CrossEncoder ומחזיר את ה-top_k הסופי.
    """
    docs_small = vs_small.similarity_search(question, k=top_k_per_index)
    docs_large = vs_large.similarity_search(question, k=top_k_per_index)

    # איחוד והסרת כפילויות תוכן
    all_docs = docs_small + docs_large
    unique_docs = []
    seen_contents = set()
    
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)

    if not unique_docs:
        return []

    # Reranking בעזרת CrossEncoder
    pairs = [[question, doc.page_content] for doc in unique_docs]
    scores = reranker_model.predict(pairs)

    doc_score_pairs = list(zip(unique_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in doc_score_pairs[:final_top_k]]


def generate_rag_response(question: str, retrieved_docs: list) -> str:
    """
    מנסח פרומפט עם הקטעים שנשלפו ומעביר ל-Claude ליצירת תשובה.
    """
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""You are an insurance assistant. Answer the user's question concisely based ONLY on the provided context below.
If the context does not contain enough information to answer the question accurately, explicitly state "I do not know".

Context:
{context_text}

Question: {question}"""

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()


def run_rag_pipeline(question: str, top_k: int = 5, top_k_per_index: int = 10) -> dict:
    """הרצת תהליך RAG מלא: שליפה מולטי-סקייל -> Reranking -> יצירת תשובה."""
    start_time = time.time()

    # שימוש בשליפה המאוחדת Multi-Scale
    retrieved_docs = retrieve_multi_scale_with_rerank(
        question, top_k_per_index=top_k_per_index, final_top_k=top_k
    )

    response_text = generate_rag_response(question, retrieved_docs)

    latency = time.time() - start_time

    return {
        "question": question,
        "response": response_text,
        "context_used": [doc.page_content for doc in retrieved_docs],
        "sources": [doc.metadata for doc in retrieved_docs],
        "latency": latency,
    }


if __name__ == "__main__":
    test_q = "What are the minimum limits of liability required by Virginia law under this auto policy?"
    print(f"Testing Multi-Scale FAISS RAG Pipeline...\nQuestion: {test_q}\n")
    
    res = run_rag_pipeline(test_q, top_k=3)
    
    print("--- Response ---")
    print(res["response"])
    print(f"\nLatency: {res['latency']:.2f}s")
    print("\n--- Top Retrieved Chunk ---")
    if res["context_used"]:
        print(res["context_used"][0][:200] + "...")