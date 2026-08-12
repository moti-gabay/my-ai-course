import sys
import os
from openai import OpenAI

# 1. קבלת נתיב לקובץ ושאלה מתוך ה-CLI
if len(sys.argv) < 3:
    print("Usage: python file_qa.py <file_path> \"<question>\"")
    sys.exit(1)

file_path = sys.argv[1]
question = sys.argv[2]

if not os.path.exists(file_path):
    print(f"Error: File {file_path} not found.")
    sys.exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    document_content = f.read()

# 2. אתחול הלקוח
client = OpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# 3. הנחיות היסוד מהשקופיות (מניעת הזיות ודרישת ציטוט מדויק)
SYSTEM_PROMPT = """
You are a strict grounded QA tool.
Rules:
1. Answer the question ONLY using the provided document. Do not use outside knowledge or guess.
2. Quote the exact passage from the document that supports your answer.
3. If the answer is not in the document, reply strictly with: "I can't find that in the document." and nothing else.
"""

USER_CONTENT = f"""
--- Document Start ---
{document_content}
--- Document End ---

Question: {question}
"""

response = client.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_CONTENT}
    ],
    temperature=0.0
)

print("\n--- Model Response ---")
print(response.choices[0].message.content)