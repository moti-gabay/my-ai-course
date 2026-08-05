import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. יצירת הלקוח בהתאם להנחיות: חיבור ל-OpenAI SDK אך הפנייה ל-Anthropic API
client = OpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

MODEL_NAME = "claude-haiku-4-5"

print("--- 1. Exercise 1: Basic Call + System Prompt ---")
# השקופיות מראות הפרדה בין System ל-User (תרגיל 1b)
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "You are a pirate culinary expert. Answer in strictly one sentence."},
        {"role": "user", "content": "What is the secret to a good pizza?"}
    ],
    temperature=0.7
)
print("Reply:", response.choices[0].message.content)


print("\n--- 2. Exercise 1b: Temperature Experiment (0 vs 1) ---")
# שקופית 38: השוואה בין determinism (0) ל-creativity (1)
for temp in [0.0, 1.0]:
    print(f"\nRunning with Temperature = {temp}:")
    for i in range(2):
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Give me one random ingredient name."}],
            temperature=temp
        )
        print(f"  Run {i+1}: {res.choices[0].message.content.strip()}")


print("\n--- 3. Exercise 1b: Structured Output (JSON & Pydantic) ---")
# שקופית 26: הגדרת JSON Schema ובניית אובייקט Pydantic
json_prompt = """
You are a culinary expert. Respond ONLY with a raw JSON object matching this schema:
{
  "answer": "string",
  "confidence": number between 0 and 1
}
Do not include markdown code block formatting (like ```json).
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
print("Raw Model Output:\n", raw_content)

# א) פענוח בעזרת json module מובנה
parsed_dict = json.loads(raw_content)
print(f"\n(a) Python Dict Parsing: Answer='{parsed_dict['answer']}', Confidence={parsed_dict['confidence']}")

# ב) אימות ומיפוי בעזרת Pydantic
class AnswerModel(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)

validated_obj = AnswerModel.model_validate_json(raw_content)
print(f"(b) Pydantic Validation: Answer='{validated_obj.answer}', Confidence={validated_obj.confidence}")