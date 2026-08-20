import os
import time
from anthropic import Anthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# 1. הגדרות נתיבים ומודלים
INDEX_DIR = "index"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# 2. טעינת מודל ה-Embeddings ואינדקס FAISS
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = FAISS.load_local(
    INDEX_DIR, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. אתחול הלקוחות והמודלים
anthropic_client = Anthropic()
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_context_with_rerank(question: str, initial_top_k: int = 20, final_top_k: int = 5) -> list:
    """
    1. שולף initial_top_k קטעים מ-FAISS.
    2. מדרג אותם מחדש בעזרת CrossEncoder.
    3. מחזיר את final_top_k הקטעים הטובים ביותר.
    """
    initial_docs = vectorstore.similarity_search(question, k=initial_top_k)
    
    if not initial_docs:
        return []

    # יצירת זוגות (שאלה, קטע) עבור ה-Reranker
    pairs = [[question, doc.page_content] for doc in initial_docs]

    # חישוב ציוני התאמה
    scores = reranker_model.predict(pairs)

    # מיון המסמכים לפי הציון הגבוה ביותר
    doc_score_pairs = list(zip(initial_docs, scores))
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


def run_rag_pipeline(
    question: str, top_k: int = 5, initial_top_k: int = 20
) -> dict:
    """הרצת תהליך RAG מלא: שליפה מורחבת -> Reranking -> יצירת תשובה."""
    start_time = time.time()

    retrieved_docs = retrieve_context_with_rerank(
        question, initial_top_k=initial_top_k, final_top_k=top_k
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
    print(f"Testing FAISS RAG Pipeline...\nQuestion: {test_q}\n")
    
    res = run_rag_pipeline(test_q, top_k=3)
    
    print("--- Response ---")
    print(res["response"])
    print(f"\nLatency: {res['latency']:.2f}s")
    print("\n--- Top Retrieved Chunk ---")
    if res["context_used"]:
        print(res["context_used"][0][:200] + "...")