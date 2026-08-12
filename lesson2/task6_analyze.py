"""
Assignment 2 — Task 6: run the judge's verdicts into decisions and validate the judge.

Three steps:
  1. Merge Length (computed in code) into the judge's four text verdicts, then apply
     the SAME pass/fail rules from Task 1 to every one of the 100 rows.
  2. Compute per-criterion agreement between the judge and the 12 rows you scored by
     hand in Task 3 (the ground truth).
  3. Print where judge and human agree vs diverge, so you can read the explanations
     on the disagreements and decide whether the fix is the judge, the rubric, or you.

Pass rule (rubric): >=3 'good' out of the 5 text/length criteria (Fluency, Grammar,
Tone, Length, Grounding), 0 'bad' among those 5, AND Grounding == 'good'.
Latency is reported separately, never part of pass/fail.

Run: python task6_analyze.py
Inputs : assignment_02_judged.xlsx (judge verdicts, 100 rows)
Output : assignment_02_final.xlsx  (adds Length, final_score, and human columns)
"""

import pandas as pd
from length_scorer import length_band

JUDGED_XLSX = "assignment_02_judged.xlsx"
OUTPUT_XLSX = "assignment_02_final.xlsx"

# Your Task 3 manual verdicts on the 12 baseline products (the ground truth).
HUMAN = {
    1:  {"Fluency": "good", "Grammar": "ok",   "Tone": "ok",   "Grounding": "bad"},
    4:  {"Fluency": "ok",   "Grammar": "good", "Tone": "bad",  "Grounding": "bad"},
    8:  {"Fluency": "good", "Grammar": "good", "Tone": "good", "Grounding": "good"},
    30: {"Fluency": "ok",   "Grammar": "ok",   "Tone": "ok",   "Grounding": "bad"},
    33: {"Fluency": "good", "Grammar": "good", "Tone": "good", "Grounding": "good"},
    36: {"Fluency": "good", "Grammar": "good", "Tone": "bad",  "Grounding": "good"},
    44: {"Fluency": "good", "Grammar": "good", "Tone": "ok",   "Grounding": "bad"},
    47: {"Fluency": "good", "Grammar": "good", "Tone": "ok",   "Grounding": "bad"},
    50: {"Fluency": "good", "Grammar": "good", "Tone": "ok",   "Grounding": "good"},
    62: {"Fluency": "ok",   "Grammar": "ok",   "Tone": "ok",   "Grounding": "good"},
    82: {"Fluency": "good", "Grammar": "good", "Tone": "ok",   "Grounding": "good"},
    84: {"Fluency": "good", "Grammar": "good", "Tone": "ok",   "Grounding": "bad"},
}

CRITERIA = ["Fluency", "Grammar", "Tone", "Grounding"]


def latency_band(ms):
    if ms is None or ms < 0:
        return ""
    if ms <= 13000:
        return "good"
    if ms <= 18000:
        return "ok"
    return "bad"


def final_score(row):
    """Pass rule over the 5 text/length criteria (Latency excluded)."""
    five = [row["judge_Fluency"], row["judge_Grammar"], row["judge_Tone"],
            row["Length"], row["judge_Grounding"]]
    if any(v not in ("good", "ok", "bad") for v in five):
        return ""                              # incomplete row
    goods = sum(v == "good" for v in five)
    bads = sum(v == "bad" for v in five)
    if row["judge_Grounding"] != "good":       # go/no-go
        return "fail"
    if bads > 0:
        return "fail"
    if goods >= 3:
        return "pass"
    return "fail"


# ---- Step 1: merge Length + Latency, compute final_score for all 100 ---------
df = pd.read_excel(JUDGED_XLSX)
df["Length"] = df["generated_description"].apply(length_band)
df["Latency"] = df["latency_ms"].apply(latency_band)
df["final_score"] = df.apply(final_score, axis=1)

pass_n = (df["final_score"] == "pass").sum()
print(f"STEP 1 — pass/fail over all 100 (judge + computed Length):")
print(f"  PASS: {pass_n}   FAIL: {(df['final_score']=='fail').sum()}   "
      f"pass rate: {pass_n/len(df)*100:.1f}%\n")

# ---- Step 2: agreement vs the 12 human-scored rows --------------------------
print("STEP 2 — judge vs human agreement (12 ground-truth rows):")
agree_counts = {c: 0 for c in CRITERIA}
rows_out = []
for pid, human in HUMAN.items():
    jr = df[df["id"] == pid].iloc[0]
    row = {"id": pid}
    for c in CRITERIA:
        j = jr[f"judge_{c}"]
        h = human[c]
        match = (j == h)
        agree_counts[c] += match
        row[f"{c}"] = f"{'=' if match else 'X'} h:{h}/j:{j}"
    rows_out.append(row)

n = len(HUMAN)
for c in CRITERIA:
    print(f"  {c:<10} {agree_counts[c]}/{n} = {agree_counts[c]/n*100:.0f}%")
overall = sum(agree_counts.values()) / (n * len(CRITERIA)) * 100
print(f"  {'OVERALL':<10} {overall:.0f}%\n")

# ---- Step 3: show the per-row detail so you can read the disagreements -------
print("STEP 3 — per-row detail (X marks a disagreement to investigate):")
detail = pd.DataFrame(rows_out).set_index("id")
print(detail.to_string())

# add human columns to the output for the 12 rows
for c in CRITERIA:
    df[f"human_{c}"] = df["id"].map(lambda i: HUMAN.get(i, {}).get(c, ""))

df.to_excel(OUTPUT_XLSX, index=False)
print(f"\nSaved -> {OUTPUT_XLSX}")
print("Read the judge_*_expl column on the X rows: is the judge wrong, were you "
      "inconsistent, or was the rubric ambiguous? That third case is the key finding.")