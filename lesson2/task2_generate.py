"""
Assignment 2 — Task 2: generate a marketing description for every product.

Runs Qwen2.5-0.5B-Instruct locally, records the text plus latency and token
counts, and writes assignment_02.xlsx with empty scoring columns for Task 3/6.

Decoding: deterministic greedy (do_sample=False). This makes the baseline
reproducible so that in Task 4 a change in the score is attributable to the
change you made, not to sampling noise.
"""

import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- config (these are your Task 4 levers — change ONE at a time) -----------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_TSV = "electrical_items_dataset.xlsx"
OUTPUT_XLSX = "assignment_02.xlsx"
MAX_NEW_TOKENS = 200          # ~90 words is ~130 tokens; headroom so Length is the model's choice, not a hard cut
CHECKPOINT_EVERY = 10         # save partial results every N products

SYSTEM_PROMPT = (
   "You are a strict e-commerce copywriter. Write a persuasive product description "
    "between 50 and 90 words based ONLY on the provided product name and attributes.\n\n"
    "CRITICAL RULES:\n"
    "1. Do NOT invent or add any technical specs, features, numbers, materials, or capabilities "
    "that are not explicitly mentioned in the source attributes.\n"
    "2. Stick strictly to a length of 50 to 90 words.\n"
    "3. Do not use hard-sell phrases like 'BUY NOW' or multiple exclamation marks.")

# ---- load model -------------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)
device = model.device
print(f"Model loaded on: {device}")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def generate_one(name: str, attributes: str):
    """Return (text, latency_ms, input_tokens, output_tokens) for one product."""
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
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                 # greedy → deterministic baseline
            pad_token_id=tokenizer.pad_token_id,
        )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    input_tokens = inputs["input_ids"].shape[1]          # includes the system prompt
    output_tokens = output_ids.shape[1] - input_tokens
    text = tokenizer.decode(
        output_ids[0][input_tokens:], skip_special_tokens=True
    ).strip()
    return text, latency_ms, input_tokens, output_tokens


# ---- warm-up (NOT measured) -------------------------------------------------
# The first call is always far slower (caches, kernels). Run one throwaway call
# so the 100 real measurements aren't skewed by cold-start cost.
print("Warm-up call (not recorded)...")
_t, warm_latency, _i, _o = generate_one(
    "Sample Product", "A simple test item with basic attributes."
)
print(f"Warm-up latency: {warm_latency} ms  "
      f"(use the real per-product times below to calibrate the Latency bands)")

# ---- main loop --------------------------------------------------------------
df = pd.read_excel(INPUT_TSV)
results = []
empty_cols = {
    "Fluency": "", "Grammar": "", "Tone": "", "Length": "",
    "Grounding": "", "Latency": "", "final_score": "",
}

for i, (_, row) in enumerate(df.iterrows(), start=1):
    try:
        text, latency_ms, in_tok, out_tok = generate_one(
            row["name"], row["description"]
        )
    except Exception as e:                       # never lose 99 rows to 1 failure
        print(f"  [{i}/{len(df)}] ERROR on id={row['id']}: {e}")
        text, latency_ms, in_tok, out_tok = f"<ERROR: {e}>", -1, -1, -1

    results.append({
        "id": row["id"],
        "name": row["name"],
        "source_attributes": row["description"],  # kept for the Task 6 judge (grounding needs both sides)
        "generated_description": text,
        "latency_ms": latency_ms,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        **empty_cols,
    })

    print(f"  [{i}/{len(df)}] {row['name'][:40]:<40} {latency_ms:>8} ms  "
          f"({out_tok} tok)")

    if i % CHECKPOINT_EVERY == 0:                 # crash-safe partial saves
        pd.DataFrame(results).to_excel(OUTPUT_XLSX, index=False)

# ---- final save -------------------------------------------------------------
pd.DataFrame(results).to_excel(OUTPUT_XLSX, index=False)

lat = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
if lat:
    lat.sort()
    print(f"\nDone. {len(results)} rows -> {OUTPUT_XLSX}")
    print(f"Latency ms  min={lat[0]}  median={lat[len(lat)//2]}  max={lat[-1]}")
    print("Use median to sanity-check your rubric's Latency bands (2000 / 5000 ms).")