from transformers import pipeline

print("Loading local Hugging Face model...")

# בשקופית 39 ובמשימה מוגדר שימוש ב-pipeline למשימת text-generation
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

prompt = "Give me a quick recipe for a gluten-free pizza crust."

# הרצת המודל מקומית על המחשב
outputs = pipe(prompt, max_new_tokens=100)

print("\n--- Local Model Output ---")
print(outputs[0]["generated_text"])