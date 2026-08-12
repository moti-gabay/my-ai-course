"""
Assignment 2 — Task 4, Experiment 3: stronger model (qwen2.5:3b via Ollama).

Hypothesis (from Experiment 2): few-shot got the 0.5B model to 83.3%, leaving two
failures that are unit-confusion errors (5 GHz -> 5 Gbps, 30 dB noise-level ->
"noise reduction"). Those look like a capability ceiling, not a prompt problem.
A model ~6x larger, given the SAME few-shot prompt, should clear them — proving
the residual failures were the model, not the prompt.

Only ONE thing changes vs Experiment 2: the model (Qwen2.5-0.5B -> qwen2.5:3b).
Same few-shot prompt, same greedy decoding (temperature 0), same 12 products.

Prereq: Ollama running with the model pulled:
    ollama pull qwen2.5:3b
    ollama serve   &      # if not already running

Run: python task4_experiment3_bigmodel.py
Output: task4_exp3_bigmodel.xlsx
"""

import time
import requests
import pandas as pd

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"   # 127.0.0.1, not localhost (WSL IPv6 quirk)
MODEL = "qwen2.5:3b"
INPUT_FILE = "electrical_items_dataset.xlsx"
OUTPUT_XLSX = "task4_exp3_bigmodel.xlsx"
SAMPLE_IDS = [1, 4, 8, 30, 33, 36, 44, 47, 50, 62, 82, 84]

# --- identical prompt to Experiment 2 (only the model changes) ---------------
SYSTEM_PROMPT = (
    "You are an e-commerce copywriter. Write one persuasive 50-90 word product "
    "description using only the given attributes. Follow the style of the examples."
)

FEWSHOT = [
    {"role": "user", "content":
        "Product: Electric Kettle\n"
        "Attributes: A concealed 2200 W element boils 1.5 litres. Auto shut-off "
        "at boiling and boil-dry protection."},
    {"role": "assistant", "content":
        "Boil water quickly and safely with this electric kettle. Its concealed "
        "2200 W element brings 1.5 litres to a rolling boil in moments, so your "
        "tea or coffee is ready when you are. Auto shut-off switches the kettle "
        "off the moment the water boils, while boil-dry protection guards it if "
        "the kettle runs empty. A dependable everyday choice for any kitchen."},
    {"role": "user", "content":
        "Product: Table Fan\n"
        "Attributes: A 45 W motor drives three speeds across a 40 cm blade span. "
        "Adjustable tilt and a stable weighted base."},
    {"role": "assistant", "content":
        "Stay cool with this quietly efficient table fan. A 45 W motor drives a "
        "40 cm blade across three speeds, from a gentle breeze to a brisk "
        "draught, so you can dial in exactly the airflow you want. Adjustable "
        "tilt directs air where you need it, and the weighted base keeps the fan "
        "steady on any desk or shelf. Simple, sturdy, and easy to live with."},
]


def generate_one(name, attributes):
    """Call Ollama /api/chat. Returns (text, latency_ms, prompt_tokens, eval_tokens)."""
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEWSHOT
        + [{"role": "user", "content": f"Product: {name}\nAttributes: {attributes}"}]
    )
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},   # greedy, to match the deterministic baseline
        "keep_alive": "30m",
    }
    start = time.perf_counter()
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    r.raise_for_status()
    data = r.json()

    text = data["message"]["content"].strip()
    # Ollama returns real token counts — the same fields the assignment wants:
    in_tok = data.get("prompt_eval_count", -1)   # tokens sent (incl. system + few-shot)
    out_tok = data.get("eval_count", -1)         # tokens generated
    return text, latency_ms, in_tok, out_tok


# sanity check that Ollama is reachable and the model is present
try:
    tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=10).json()
    names = [m["name"] for m in tags.get("models", [])]
    if not any(MODEL in n for n in names):
        print(f"WARNING: '{MODEL}' not found in Ollama. Pulled models: {names}")
        print(f"Run:  ollama pull {MODEL}")
except Exception as e:
    print(f"Cannot reach Ollama at 127.0.0.1:11434 — is 'ollama serve' running? ({e})")

print("Warm-up...")
generate_one("Sample Product", "A simple test item.")

df = pd.read_excel(INPUT_FILE)
sample = df[df["id"].isin(SAMPLE_IDS)]

results = []
for _, row in sample.iterrows():
    try:
        text, latency_ms, in_tok, out_tok = generate_one(row["name"], row["description"])
    except Exception as e:
        print(f"  ERROR id={row['id']}: {e}")
        text, latency_ms, in_tok, out_tok = f"<ERROR: {e}>", -1, -1, -1
    word_count = len(text.split())
    results.append({
        "id": row["id"], "name": row["name"],
        "source_attributes": row["description"],
        "generated_description": text,
        "word_count": word_count,
        "latency_ms": latency_ms, "input_tokens": in_tok, "output_tokens": out_tok,
        "Fluency": "", "Grammar": "", "Tone": "", "Length": "",
        "Grounding": "", "Latency": "", "final_score": "",
    })
    print(f"  id={row['id']:<3} {row['name'][:32]:<32} "
          f"{word_count:>3}w  {latency_ms:>8}ms")

pd.DataFrame(results).to_excel(OUTPUT_XLSX, index=False)
print(f"\nDone -> {OUTPUT_XLSX}")
print("Score by hand, then compare to Exp 2 (83.3%). Watch #30 and #82 — did the "
      "bigger model fix the unit-confusion errors (GHz/Gbps, dB)?")
print("Note: latency will differ from the transformers runs — Ollama + 3B is a "
      "different engine and size, so compare pass rate and grounding, not raw ms.")