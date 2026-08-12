# Assignment 2 — Evaluation-Driven Development · Submission

**Domain:** e-commerce electronics catalogue (100 products). The model turns each
product's `name` + technical `description` (the source attributes) into a persuasive
50–90 word marketing description. Generator: Qwen2.5-0.5B-Instruct (local, CPU).
Judge: Claude Sonnet (different model family, to avoid self-enhancement bias).

Repo files: `rubric.md`, `task2_generate.py`, `task4_experiment1_prompt.py`,
`task4_experiment2_fewshot.py`, `task4_experiment3_bigmodel.py`, `length_scorer.py`,
`task5_judge.py`, `task6_analyze.py`, `assignment_02.xlsx` (+ judged/final variants).

---

## Task 1 — Rubric

Full definitions in `rubric.md`. Summary of the 18 bands (6 criteria × 3):

| Criterion | good | ok | bad |
|-----------|------|----|----|
| Fluency | no awkward phrasing | 1–2 clunky phrases | 3+ awkward, or must re-read |
| Grammar | zero errors | 1–2 minor errors | 3+ errors, or one that obscures meaning |
| Tone | warm & persuasive throughout | mostly on-voice, 1–2 flat spots | dry/robotic, or salesy (`!!`, ALL CAPS, "BUY NOW") |
| Length | 50–90 words | 40–49 or 91–110 | <40 or >110 |
| Grounding | all facts supported | spec-based exaggeration, no new fact | ≥1 invented fact |
| Latency | ≤13000 ms | 13001–18000 ms | >18000 ms |

**Grounding ruling (lenient):** generic filler ("perfect for everyday use") is
allowed; only an invented *fact* (a spec/feature/dimension not in the source) is a
failure. `ok` is reserved for subjective exaggeration built on a real spec.

**Pass rule:** ≥3 `good` of the 5 text/length criteria (Latency excluded), 0 `bad`
among those 5, **and** Grounding = `good` (go/no-go). Grounding is the go/no-go
because a fluent lie ("waterproof" when it isn't) is worse than a clumsy truth.

**Latency** was recalibrated to the measured CPU distribution (median ~15000 ms);
the original 2000/5000 ms bands would have marked every row `bad`.

---

## Task 3 — Human baseline (12 products)

Scored 12 products by hand on the Task 2 baseline output. Pass rate: **33.3%
(4/12)**.

**Error analysis — failures are clustered, not scattered:** 6/12 failed the
Grounding go/no-go. The 0.5B model tries to "enrich" the copy and invents technical
specs absent from the input — e.g. an RTX 3080 GPU (#44), a 32 GB DDR4 cache (#47),
a "W/sq inch" unit (#1). Fluency and Grammar were mostly fine. So the target for
improvement was clear: **Grounding first**, then Length and Tone. This clustering is
what made the improvement cycle targeted rather than a guess.

---

## Task 4 — Improvement cycle (3 experiments)

Baseline pass rate: **33.3%**. Each experiment changed **one** thing and was
re-scored on the same 12 products with the same rubric.

| Metric | Baseline (0.5B) | Exp 1 (0.5B + rules) | Exp 2 (0.5B + few-shot) | Exp 3 (stronger model) |
|--------|-----------------|----------------------|-------------------------|------------------------|
| Pass rate | 33.3% (4/12) | 8.3% (1/12) | 83.3% (10/12) | 91.7% (11/12) |
| Grounding failures | 50% | 58% | 16.7% | 8.3% |
| Length violations | 25% | 66.7% | 0% | 0% |
| Avg input tokens | ~150 | ~300 | ~365 | ~365 |

**Experiment 1 — prompt hardening (no few-shot).**
*Change:* a long system prompt with explicit negative rules (no invention, no
hard-sell, strict length) + a self-check.
*Hypothesis:* explicit constraints will fix the 50% Grounding failures.
*Result:* pass rate **dropped to 8.3%**. Verbal rules did remove specific
hallucinations (the RTX 3080, the DDR4 cache, the W/sq inch unit all disappeared),
but a 0.5B model can't absorb a long rule-set: it produced **instruction echo**
(printing the rule text and "Hard Rules" headers into the output, #8/#33/#36/#44) and
**bloated the length** (up to 135 words), while inventing *new* facts to compensate
(#4 "70% saving", #50 "500 mm"). Lesson: more prompting can make a small model worse.

**Experiment 2 — few-shot (in-context learning).**
*Change:* simplified the prompt back to one line and added two worked examples
(good, grounded, correctly sized).
*Hypothesis:* showing the pattern beats telling the rules, and avoids instruction echo.
*Result:* pass rate **jumped to 83.3%**. Length violations went to **0%** and
instruction echo **vanished** — the model mimicked the examples' structure and
length instead of parsing rules. Two Grounding failures remained (#30, #82), both
**unit-confusion** errors (5 GHz → 5 Gbps; a 30 dB noise *level* → "noise
reduction") — a capability ceiling, not a prompt problem.
*Cost side of the trade-off:* input tokens rose ~2.4× (150 → 365) because the
examples are re-sent on every call. We paid tokens to buy pass rate.

**Experiment 3 — stronger model (qwen2.5:3b), same few-shot prompt.**
*Change:* only the model (0.5B → 3B); prompt held identical to Exp 2.
*Hypothesis:* the residual unit-confusion errors are a capability limit a larger
model will clear.
*Result:* pass rate **91.7%**. Both unit-confusion failures were fixed. One new
failure survived — #50 invented physical dimensions ("10 cm × 18 cm"). That single
residual is the justification for the go/no-go and for Task 5: **even a stronger
model needs an automated quality gate.**

**Cross-experiment lesson:** the three levers do different things. Prompt-hardening
hurt a small model; in-context learning fixed format and tone without a model change;
model-scaling cleared the final capability errors. Length dropping to 0% at Exp 2 and
staying there proves few-shot (not model size) fixed length.

---

## Task 5 — Judge design: why `explanation` before `verdict`

The Pydantic schema returns, per criterion, `explanation` **then** `verdict`. An
autoregressive model generates tokens in order, each conditioned on everything
already emitted. With the explanation first, the verdict token is produced *after*
the reasoning and is conditioned on it — genuine chain-of-thought. Reverse the order
and the verdict is committed first, then the model writes a rationalisation for a
conclusion it already fixed; the reasoning can no longer influence the verdict.
Ordering the fields is therefore what makes the reasoning actually do work.

Length is computed in code (`length_scorer.py`, `len(text.split())`) and Latency is
measured — neither is sent to the judge. The judge scores only the four text
criteria, and for Grounding receives both the source attributes and the generated
text, because grounding is a comparison and needs both sides.

---

## Task 6 — Judge analysis & recommendation

Full analysis in `task6_writeup.md`. Headline results:

**Agreement (12 ground-truth rows):** Grammar 67% · Tone 58% · Grounding 58% ·
Fluency 25% · **overall 52%**. The countable criterion agreed most; the subjective
ones diverged most — as predicted — and the divergence has a consistent *direction*
(the judge is harsher on Fluency/Grounding, more lenient on Tone), so it's a
biased-but-consistent instrument, not a noisy one.

**Three kinds of disagreement:**
- *Judge right, human missed it* (Grounding #8, #82): the judge caught subtle
  distortions — "control dough temperature" (source only has a safety cut-out),
  "three nodes share one 5 GHz radio" (source reserves it for backhaul). The judge
  *adds* value on the criterion that carries legal risk.
- *Rubric ambiguous* (Fluency #1, #36): "1–2 clunky phrases = ok" never says whether
  *clunky* means bad syntax or strange content, so two reasonable readers split.
  **The most valuable finding — the fix is to tighten the rubric, not blame either
  evaluator.**
- *Systematic bias* (Tone): human `ok` → judge `good` repeatedly; the judge reads
  competent copy as fully on-voice.

**Trade-offs:** the human is the accuracy anchor but inconsistent and unscalable; the
judge is cheap, fast, scalable, and repeatable (temperature 0) — including repeatably
biased. Much of the 52% gap is the ambiguous Fluency band, which would inflate once
fixed.

**Recommendation (not a single choice):** automate the bulk with the judge; keep
Grounding as an enforced go/no-go scored by the judge with the source always
supplied (the judge is stricter here — the safe direction); put humans only at
escalation points (audit Grounding-`bad` rows, sample daily for drift, trigger a
rubric review when agreement drops). **Fix the rubric before any model change** — an
ambiguous rubric doesn't get more accurate when automated; it just gets applied at
scale, silently.

---

## The takeaway

The durable lesson: the evaluation is only as good as the rubric. Task 1 looked easy
and turned out to be the hardest, most important work — and Task 6 is where its
ambiguities surfaced, at scale, in the 25% Fluency agreement. The full EDD loop
(33.3% → 8.3% → 83.3% → 91.7%) only produced trustworthy numbers because the rubric
was written first and applied identically by hand and by the judge.