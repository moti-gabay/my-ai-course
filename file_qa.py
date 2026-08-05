import sys
import os
from openai import OpenAI

# 1. בדיקת ארגומנטים בשורת הפקודה
if len(sys.argv) < 3:
    print("שימוש: python file_qa.py <path_to_file> \"<question>\"")
    sys.exit(1)

file_path = sys.argv[1]
question = sys.argv[2]

# 2. קריאת תוכן הקובץ
if not os.path.exists(file_path):
    print(f"שגיאה: הקובץ {file_path} לא נמצא.")
    sys.exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    document_content = f.read()

# 3. חיבור ל-Claude
client = OpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

SYSTEM_PROMPT = """
You are a strict QA assistant.
Rules:
1. Answer the user's question ONLY using the provided document content. Do not use outside knowledge or guess.
2. You MUST quote the exact passage from the document that supports your answer.
3. If the answer cannot be found in the document, reply strictly with: "I can't find that in the document." and nothing else.
"""

USER_CONTENT = f"""
Document:
{document_content}

Question:
{question}
"""
MODEL_NAME = "claude-haiku-4-5"

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_CONTENT}
    ],
    temperature=0.0
)

print("\n--- תשובת המערכת ---")
print(response.choices[0].message.content)