import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. יצירת הלקוח - פנייה ל-Claude באמצעות כתובת ה-API המתאימה של Anthropic
client = OpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

MODEL_NAME = "claude-haiku-4-5"
print("--- 1. קריאה בסיסית + System Prompt ---")
# הודעת System מגדירה אישיות/חוקים, הודעת User היא השאלה
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "You are a friendly recipes expert. Answer in strictly one sentence."},
        {"role": "user", "content": "What is the secret to a good pizza?"}
    ],
    temperature=0.7
)
print("תשובת המודל:", response.choices[0].message.content)


print("\n--- 2. ניסוי Temperature (0 לעומת 1) ---")
for temp in [0.0, 1.0]:
    print(f"\nהרצה עם Temperature = {temp}:")
    for i in range(2):
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Give me one random ingredient name."}],
            temperature=temp
        )
        print(f"  הרצה {i+1}: {res.choices[0].message.content.strip()}")


print("\n--- 3. פלט מובנה (Structured Output - JSON) ---")
# הדרכת המודל להחזיר רק JSON
json_prompt = """
You are a culinary expert. Respond ONLY with a raw JSON object matching this schema:
{
  "answer": "string",
  "confidence": number between 0 and 1
}
Do not include markdown formatting like ```json.
"""

res_json = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": json_prompt},
        {"role": "user", "content": "Can I replace butter with oil in cakes?"}
    ],
    temperature=0.2
)
raw_content = res_json.choices[0].message.content.strip()
print("הטקסט הגולמי מהמודל:\n", raw_content)

# דרך א': פענוח בעזרת json מובנה ב-Python
parsed_dict = json.loads(raw_content)
print(f"\n(א) חילוץ מ-dict: תשובה='{parsed_dict['answer']}', ביטחון={parsed_dict['confidence']}")

# דרך ב': אימות ומיפוי בעזרת Pydantic
class RecipeAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)

validated_obj = RecipeAnswer.model_validate_json(raw_content)
print(f"(ב) אובייקט Pydantic מאומת: answer='{validated_obj.answer}', confidence={validated_obj.confidence}")