"""
Assignment 2 — Task 4, Experiment 2: few-shot prompting.

Hypothesis (from Experiment 1): a 0.5B model can't follow a long verbal rule-set
— it echoed the rules into the output and bloated the length. Instead of TELLING
it the rules, SHOW it: a simplified system prompt plus two worked examples (one
good, one bad-then-corrected) should teach length, grounding, and no-hard-sell
implicitly, via in-context learning, without triggering instruction echo.

Only ONE thing changes vs Experiment 1: the prompt strategy (short prompt +
examples instead of a long rule list). Same model, same greedy decoding, same 12
products.

Run: python task4_experiment2_fewshot.py
Output: task4_exp2_fewshot.xlsx
"""

import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_FILE = "electrical_items_dataset.xlsx"
OUTPUT_XLSX = "task4_exp2_fewshot.xlsx"
MAX_NEW_TOKENS = 200
SAMPLE_IDS = [1, 4, 8, 30, 33, 36, 44, 47, 50, 62, 82, 84]

# Simplified system prompt — short, to avoid the instruction echo Exp 1 caused.
# The examples below carry the rules instead of a long verbal list.
SYSTEM_PROMPT = (
    "You are an e-commerce copywriter. Write one persuasive 50-90 word product "
    "description using only the given attributes. Follow the style of the examples."
)

# Two worked examples delivered as prior chat turns (in-context learning).
# Example 1: a clean, grounded, correctly-sized description (what GOOD looks like).
# Example 2: the same product done BADLY, then the corrected version — showing the
# model the difference between inventing facts / hard-sell and staying grounded.
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
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEWSHOT
        + [{"role": "user", "content": f"Product: {name}\nAttributes: {attributes}"}]
    )
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


print("Warm-up...")
generate_one("Sample Product", "A simple test item.")

df = pd.read_excel(INPUT_FILE)
sample = df[df["id"].isin(SAMPLE_IDS)]

results = []
for _, row in sample.iterrows():
    text, latency_ms, in_tok, out_tok = generate_one(row["name"], row["description"])
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
print("Note: few-shot raises input_tokens (the examples are re-sent every call) — "
      "that's the cost side of the trade-off; record it against the pass-rate gain.")