from langchain_huggingface import HuggingFaceEmbeddings

print("⏳ טוען את מודל ה-Embeddings המקומי (bge-small-en-v1.5)...")

# טעינת המודל המקומי
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# בדיקת המרה קצרה
query = "Represent this sentence for searching relevant passages: test query"
vector = embeddings.embed_query(query)

print(f"✅ המודל נטען בהצלחה! אורך ה-Vector שהתקבל: {len(vector)}")