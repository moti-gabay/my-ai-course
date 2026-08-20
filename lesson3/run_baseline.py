import json
import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

EVAL_SET_PATH = "data/eval_set.json"
OUTPUT_PATH = "data/baseline_results.json"

# פונקציית "המודל השופט" לסיווג התשובה
def classify_answer(question, ground_truth, generated_answer):
    judge_prompt = f"""
    You are an expert evaluator. Compare the Generated Answer to the Ground Truth for the given Question.
    Classify the Generated Answer into EXACTLY ONE of the following 3 categories:
    - "Refused": If the generated answer explicitly states it does not know, cannot answer, or lacks context.
    - "Answered correctly": If the generated answer is factually aligned and consistent with the Ground Truth.
    - "Hallucinated": If the generated answer provides specific details that contradict the Ground Truth, or invents facts not present in the Ground Truth.

    Question: {question}
    Ground Truth: {ground_truth}
    Generated Answer: {generated_answer}

    Return ONLY the category name ("Refused", "Answered correctly", or "Hallucinated") and nothing else.
    """
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001", # למשימת שופט פשוטה כזו, Haiku מספיק ומהיר
        max_tokens=15,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    return response.content[0].text.strip()

def main():
    # 1. טעינת השאלות
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    results = []
    print(f"Starting baseline evaluation on {len(eval_data)} questions...\n")

    # 2. לולאת הרצה
    for entry in eval_data:
        q_id = entry.get("id") or entry.get("question_id", "Unknown")
        question = entry["question"]
        ground_truth = entry.get("reference_answer") or entry.get("ground_truth_answer", "")        
        # פרומפט הבסיס (ללא הקשר מוקדם)
        prompt = f"You are an insurance assistant. Answer the following question concisely based on your general knowledge. If you do not know the answer, explicitly state 'I do not know'.\n\nQuestion: {question}"
        
        # --- תחילת מדידת זמן ---
        start_time = time.time()
        
        # קריאה למודל לייצור התשובה
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # --- סיום מדידת זמן ---
        end_time = time.time()
        
        # חישוב נתונים
        latency = round(end_time - start_time, 2)
        generated_answer = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        
        # קריאה למודל השופט כדי לסווג את התשובה
        classification = classify_answer(question, ground_truth, generated_answer)
        
        # שמירת הנתונים המורחבים לאובייקט התוצאה
        results.append({
            "question_id": q_id,
            "question": question,
            "ground_truth_answer": ground_truth,
            "generated_answer": generated_answer,
            "metrics": {
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "classification": classification
        })
        
        print(f"Q: {q_id} | Latency: {latency}s | Tokens: {total_tokens} | Category: [{classification}]")

    # 3. שמירת התוצאות לקובץ JSON
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBaseline run finished! Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()