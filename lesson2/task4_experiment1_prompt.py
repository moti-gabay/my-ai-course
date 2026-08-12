"""
Assignment 2 — Task 4, Experiment 1: prompt engineering (no few-shot).

Hypothesis (from the Task 3 error analysis): 50% of products failed on Grounding
because the 0.5B model invents specs. A tightened system prompt that explicitly
forbids adding any number/feature not in the input — plus a hard-sell ban (fixes
#36) and a strict length rule (fixes #62, #30) — should lift the pass rate above
the 33.3% baseline WITHOUT changing the model.

Only ONE thing changes vs Task 2: the system prompt. Same model (Qwen2.5-0.5B),
same greedy decoding, same 12 products. That makes the delta attributable.

Run: python task4_experiment1_prompt.py
Output: task4_exp1_prompt.xlsx  (12 rows, empty score columns to fill by hand)
"""

import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_FILE = "electrical_items_dataset.xlsx"
OUTPUT_XLSX = "task4_exp1_prompt.xlsx"
MAX_NEW_TOKENS = 200

# The 12 products scored by hand in Task 3 (the baseline sample).
SAMPLE_IDS = [1, 4, 8, 30, 33, 36, 44, 47, 50, 62, 82, 84]

# --- the ONLY change vs Task 2: a tightened prompt -------------------------
# Baseline prompt for reference (Task 2):
#   "...based ONLY on the provided product name and attributes. Do not invent
#    any facts, specifications, features, or connectivity that are not present
#    in the input. Output only the description..."
# Experiment 1 hardens the three failure modes the baseline showed.
SYSTEM_PROMPT = (
    "You are an expert e-commerce copywriter. Write ONE persuasive product "
    "description from the product name and attributes below.\n\n"
    "HARD RULES — follow every one:\n"
    "1. GROUNDING: Use ONLY facts, numbers, materials, and features that appear "
    "in the attributes. Do NOT add any spec, component, brand, standard, or "
    "capability that is not written there. If a detail is not in the input, it "
    "does not exist. Inventing even one fact makes the description unusable.\n"
    "2. LENGTH: Write between 50 and 90 words. Never exceed 90. Finish your last "
    "sentence — never stop mid-sentence.\n"
    "3. TONE: Warm and credible. No hard-sell. Do NOT use exclamation marks, "
    "ALL-CAPS words, or phrases like 'Order now', 'Buy today', or 'Limited offer'.\n"
    "4. FORMAT: Output only the description itself — no title, no preamble, no "
    "quotation marks.\n\n"
    "Before finishing, silently check: is every fact in my text present in the "
    "attributes? Am I within 90 words? Did I avoid hard-sell?"
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)
device = model.device
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded on: {device}")


def generate_one(name, attributes):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Product: {name}\nAttributes: {attributes}"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    in_tok = inputs["input_ids"].shape[1]
    out_tok = output_ids.shape[1] - in_tok
    text = tokenizer.decode(output_ids[0][in_tok:], skip_special_tokens=True).strip()
    return text, latency_ms, in_tok, out_tok


# warm-up (not recorded)
print("Warm-up...")
generate_one("Sample Product", "A simple test item.")

df = pd.read_excel(INPUT_FILE)
sample = df[df["id"].isin(SAMPLE_IDS)]

results = []
for _, row in sample.iterrows():
    text, latency_ms, in_tok, out_tok = generate_one(row["name"], row["description"])
    word_count = len(text.split())          # Length is computed, per the rubric
    results.append({
        "id": row["id"],
        "name": row["name"],
        "source_attributes": row["description"],
        "generated_description": text,
        "word_count": word_count,
        "latency_ms": latency_ms,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "Fluency": "", "Grammar": "", "Tone": "", "Length": "",
        "Grounding": "", "Latency": "", "final_score": "",
    })
    print(f"  id={row['id']:<3} {row['name'][:32]:<32} "
          f"{word_count:>3}w  {latency_ms:>8}ms")

pd.DataFrame(results).to_excel(OUTPUT_XLSX, index=False)
print(f"\nDone -> {OUTPUT_XLSX}")
print("Score these 12 by hand with the SAME rubric, then compare pass rate to the "
      "33.3% baseline (4/12).")