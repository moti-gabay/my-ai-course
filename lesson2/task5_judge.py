"""
Assignment 2 — Task 5: LLM-as-a-judge (Claude Sonnet).

Automates the manual scoring from Task 3. A different model family from the
generator (Qwen) — this avoids self-enhancement bias, where a judge favours its
own family's outputs.

The judge scores ONLY the subjective text criteria: Fluency, Grammar, Tone,
Grounding. Length is computed in code (length_scorer.py) and Latency is measured —
neither is sent to the judge (see the rubric's justification).

Schema note (Task 5): for every criterion, `explanation` comes BEFORE `verdict`.
An autoregressive model generates tokens in order, each conditioned on what came
before. With explanation first, the verdict token is produced *after* — and
therefore conditioned on — the reasoning. Reverse the order and the verdict is
committed first, then the model rationalises it: chain-of-thought only helps when
the reasoning precedes the conclusion.

Grounding gets BOTH the source attributes and the generated text, because grounding
is a comparison — the judge can't detect an invented fact without the source.

Setup:
    pip install anthropic pydantic
    export ANTHROPIC_API_KEY=sk-ant-...

Run a 5-row sanity check first:
    python task5_judge.py --sanity
Then the full run in Task 6:
    python task5_judge.py --all
"""

import argparse
import json
import pandas as pd
from pydantic import BaseModel, ValidationError
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"          # different family from the Qwen generator
INPUT_XLSX = "assignment_02.xlsx"    # the 100-row Task 2 output
OUTPUT_XLSX = "assignment_02_judged.xlsx"

client = Anthropic()   # reads ANTHROPIC_API_KEY from the environment


# --- Pydantic schema: explanation BEFORE verdict, per criterion --------------
class CriterionScore(BaseModel):
    explanation: str
    verdict: str        # "good" | "ok" | "bad"


class JudgeResult(BaseModel):
    fluency: CriterionScore
    grammar: CriterionScore
    tone: CriterionScore
    grounding: CriterionScore


# --- the rubric, verbatim, so the judge applies YOUR standard ----------------
RUBRIC = """\
Score each criterion as exactly one of: good / ok / bad.

FLUENCY — do the sentences read naturally, like a human wrote them?
  good: No awkward phrasing; reads cleanly in one pass.
  ok:   1-2 clunky phrases, but the meaning is always clear.
  bad:  3+ awkward phrases, or a sentence you must re-read to parse.

GRAMMAR — spelling, punctuation, agreement.
  good: Zero errors.
  ok:   1-2 minor errors that don't impede understanding.
  bad:  3+ errors, or a single error that changes or obscures the meaning.

TONE — friendly, credible sales voice.
  good: Persuasive and warm throughout, without sounding cheap or pushy.
  ok:   Mostly on-voice, with 1-2 flat or slightly overhyped spots.
  bad:  Dry/robotic, or aggressively salesy — any of: multiple exclamation
        marks, ALL-CAPS words for emphasis, or hard-sell phrases ("BUY NOW",
        "LIMITED OFFER", "ORDER TODAY").

GROUNDING — faithful to the SOURCE ATTRIBUTES (the safety criterion).
  good: Every factual claim is supported by the source. Generic marketing filler
        ("perfect for everyday use") is fine.
  ok:   Subjective exaggeration built on a real spec (e.g. "exceptionally
        powerful" for a stated 2200 W), inventing no new fact.
  bad:  At least one invented fact: a spec, connection, material, dimension, or
        feature not present in the source attributes.
"""

SYSTEM_PROMPT = (
    "You are a meticulous evaluation judge for e-commerce product descriptions. "
    "You apply the given rubric exactly as written, not your own taste. For each "
    "criterion you first write a short explanation grounded in the rubric, then "
    "give a verdict of good, ok, or bad. For GROUNDING you compare the description "
    "only against the SOURCE ATTRIBUTES provided — any fact not in the source is an "
    "invention, even if it is plausibly true in the real world.\n\n"
    + RUBRIC
)


def judge_one(source_attributes: str, description: str) -> JudgeResult:
    """Score one description with Claude Sonnet. Returns a validated JudgeResult."""
    user = (
        f"SOURCE ATTRIBUTES (the only ground truth for Grounding):\n"
        f"{source_attributes}\n\n"
        f"GENERATED DESCRIPTION (score this):\n{description}\n\n"
        f"Return a JSON object with keys fluency, grammar, tone, grounding. "
        f"Each maps to an object with 'explanation' (string) THEN 'verdict' "
        f"(one of good/ok/bad). Output only the JSON, no prose around it."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text.strip()
    # strip accidental code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    data = json.loads(raw)
    return JudgeResult(**data)   # Pydantic raises if the shape is wrong


def run(sample_only: bool):
    df = pd.read_excel(INPUT_XLSX)
    if sample_only:
        df = df.head(5)
        print("SANITY CHECK — 5 rows. Read the explanations carefully.\n")

    judge_cols = []
    for crit in ["Fluency", "Grammar", "Tone", "Grounding"]:
        judge_cols += [f"judge_{crit}", f"judge_{crit}_expl"]

    # Resume: if a saved judged file exists, load it and skip rows already done.
    done_ids = set()
    if not sample_only:
        try:
            prev = pd.read_excel(OUTPUT_XLSX)
            done_ids = set(prev.loc[prev["judge_Grounding"].astype(str) != "", "id"])
            df = prev                                  # keep prior verdicts
            print(f"Resuming — {len(done_ids)} rows already judged, skipping them.")
        except FileNotFoundError:
            pass

    for col in judge_cols:
        if col not in df.columns:
            df[col] = ""

    processed = 0
    for i, row in df.iterrows():
        if not sample_only and row["id"] in done_ids:
            continue                                   # already judged in a prior run
        try:
            result = judge_one(row["source_attributes"], row["generated_description"])
        except (ValidationError, json.JSONDecodeError, KeyError) as e:
            print(f"  id={row.get('id','?')} PARSE/VALIDATION ERROR: {e}")
            continue
        except Exception as e:                         # API/network error — don't lose the run
            print(f"  id={row.get('id','?')} API ERROR: {e} — saving progress and stopping.")
            if not sample_only:
                df.to_excel(OUTPUT_XLSX, index=False)
            raise

        for crit, score in [("Fluency", result.fluency), ("Grammar", result.grammar),
                            ("Tone", result.tone), ("Grounding", result.grounding)]:
            df.at[i, f"judge_{crit}"] = score.verdict
            df.at[i, f"judge_{crit}_expl"] = score.explanation

        processed += 1
        if sample_only:
            print(f"id={row['id']}  {row['name']}")
            print(f"  Fluency={result.fluency.verdict}  Grammar={result.grammar.verdict}"
                  f"  Tone={result.tone.verdict}  Grounding={result.grounding.verdict}")
            print(f"  grounding reasoning: {result.grounding.explanation[:160]}\n")
        else:
            print(f"  id={row['id']:<3} judged "
                  f"F={result.fluency.verdict} G={result.grammar.verdict} "
                  f"T={result.tone.verdict} Gr={result.grounding.verdict}")
            if processed % 10 == 0:                    # checkpoint every 10 rows
                df.to_excel(OUTPUT_XLSX, index=False)
                print(f"    ...checkpoint saved ({processed} new this run)")

    if not sample_only:
        df.to_excel(OUTPUT_XLSX, index=False)
        print(f"\nDone -> {OUTPUT_XLSX}")
        print("Next (Task 6): merge Length via length_scorer, apply pass/fail, "
              "compare judge vs your 12 manual rows.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="run on 5 rows and print")
    ap.add_argument("--all", action="store_true", help="run on all rows, save file")
    args = ap.parse_args()
    if not (args.sanity or args.all):
        print("Pass --sanity (5 rows, printed) or --all (full run, saved).")
    else:
        run(sample_only=args.sanity)