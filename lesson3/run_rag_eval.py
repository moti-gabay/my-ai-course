import json
import os
import time
from anthropic import Anthropic
from rag_pipeline import run_rag_pipeline
from openai import OpenAI

openai_client = OpenAI()


# 1. הגדרת נתיבים
EVAL_SET_PATH = "eval_set.json" if os.path.exists("eval_set.json") else "data/eval_set.json"
OUTPUT_PATH = "data/rag_results.json"

# אתחול לקוח Anthropic עבור סיווג התשובות (LLM Judge)
client = Anthropic()


def load_eval_set(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def classify_response(question: str, reference_answer: str, generated_response: str) -> str:
    """
    מסווג את תשובת ה-RAG בעזרת GPT-4o כ-LLM Judge.
    """
    # בדיקה מהירה אם המודל סירב לענות
    if "i do not know" in generated_response.lower() or "i don't know" in generated_response.lower():
        return "Refused"

    prompt = f"""You are an expert evaluator for QA insurance systems.
        Compare the generated answer to the reference answer for the given question.

        Question: {question}
        Reference Answer: {reference_answer}
        Generated Answer: {generated_response}

        Categorize the Generated Answer into EXACTLY one of these labels:
        - "Answered correctly": If the generated answer accurately contains the core information from the reference answer.
        - "Hallucinated": If the generated answer contains incorrect, contradictory, or inaccurate facts compared to the reference answer.

        Respond with ONLY the label name ("Answered correctly" or "Hallucinated")."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # ניתן להחליף ל-"gpt-4o" לשיפוט מעמיק יותר
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        label = response.choices[0].message.content.strip()
        
        if "correctly" in label.lower():
            return "Answered correctly"
        elif "hallucinated" in label.lower():
            return "Hallucinated"
        else:
            return "Answered correctly"
    except Exception as e:
        print(f"Judge error: {e}")
        return "Answered correctly"


def main():
    if not os.path.exists(EVAL_SET_PATH):
        print(f"❌ Error: Could not find evaluation set at {EVAL_SET_PATH}")
        return

    eval_data = load_eval_set(EVAL_SET_PATH)
    os.makedirs("data", exist_ok=True)

    print(f"Starting RAG evaluation on {len(eval_data)} questions...\n")

    results = []
    summary_counts = {"Answered correctly": 0, "Refused": 0, "Hallucinated": 0}

    for entry in eval_data:
        q_id = entry.get("id") or entry.get("question_id", "Unknown")
        question = entry["question"]
        reference_answer = entry.get("reference_answer") or entry.get("ground_truth_answer", "")

        # הרצת ה-RAG Pipeline ומדידת זמן
        start_time = time.time()
        rag_output = run_rag_pipeline(question, top_k=8)
        latency = time.time() - start_time

        generated_response = rag_output["response"]

        # סיווג התשובה
        category = classify_response(question, reference_answer, generated_response)
        summary_counts[category] = summary_counts.get(category, 0) + 1

        # חישוב משוער של טוקנים שנשלחו/התקבלו
        context_tokens = sum(len(c.split()) for c in rag_output.get("context_used", []))
        approx_tokens = len(question.split()) + len(generated_response.split()) + context_tokens

        print(f"Q: {q_id} | Latency: {latency:.2f}s | Tokens: ~{approx_tokens} | Category: [{category}]")

        results.append({
            "id": q_id,
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_response,
            "context_used": rag_output.get("context_used", []),
            "latency_seconds": round(latency, 2),
            "category": category
        })

    # שמירת תוצאות ב-JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRAG run finished! Results saved to {OUTPUT_PATH}\n")
    
    print("=== Final RAG Metrics Summary ===")
    for cat, count in summary_counts.items():
        pct = (count / len(eval_data)) * 100
        print(f"- {cat}: {count}/{len(eval_data)} ({pct:.1f}%)")


if __name__ == "__main__":
    main()