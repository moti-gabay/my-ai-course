# Assignment 2 — Rubric (Task 1)

**Task:** generate a persuasive 50–90 word marketing description for each product, from its
`name` and technical `description` (the source attributes).
**Domain:** e-commerce electronics catalogue (100 products, English).
**This rubric is the spec.** It is written *before* any output is generated, and the exact
same definitions and pass/fail rules are applied in three places:
Task 3 (human scoring), Task 4 (re-scoring after each improvement), and Task 6 (the LLM judge).

---

## Grounding decision (read this first)

The dataset's `description` field is the **source of truth**. Every factual claim in a
generated description is checked against it.

**Ruling — the lenient stance:** generic marketing filler that makes no factual claim is
**allowed**. Only a *factual* claim absent from the source counts as an invention. Between
those two extremes sits a middle band — subjective exaggeration that leans on a real spec
without inventing a new one — and that is what separates `ok` from `good`.

The three bands, concretely:

- **good** — every factual claim is supported by the source; generic filler ("perfect for
  everyday use", "a great addition to any kitchen") is completely fine.
- **ok** — contains subjective exaggeration grounded in a real spec, but invents no new fact.
  E.g. calling a 2200 W element "an exceptionally powerful motor" — the wattage is real, the
  "exceptionally" is puffery hanging off it. No new number, feature, or capability is added.
- **bad** — at least one invented fact: a spec, connection, material, or feature not in the
  source — e.g. "connects to your phone" on a product whose source lists no wireless capability.

Rationale: real marketing copy is full of both filler and puffery, so `good` and `ok` capture
the two shades of "went beyond the source without lying". The failure that actually matters —
and that can get a retailer sued — is inventing a capability the product doesn't have, and only
that lands in `bad`.

This ruling is applied consistently across every criterion below and in every task.

---

## The six criteria — rating bands

### Fluency — do the sentences read naturally, like a human wrote them?
| Rating | Definition |
|--------|------------|
| good | No awkward phrasing; reads cleanly in one pass. |
| ok | 1–2 clunky phrases, but the meaning is always clear. |
| bad | 3+ awkward phrases, **or** a sentence you must re-read to parse. |

### Grammar — spelling, punctuation, agreement
| Rating | Definition |
|--------|------------|
| good | Zero errors. |
| ok | 1–2 minor errors that don't impede understanding. |
| bad | 3+ errors, **or** a single error that changes or obscures the meaning. |

### Tone — friendly, credible sales voice
| Rating | Definition |
|--------|------------|
| good | Persuasive and warm throughout, without sounding cheap or pushy. |
| ok | Mostly on-voice, with 1–2 flat or slightly overhyped spots. |
| bad | Dry/robotic, **or** aggressively salesy — any of: multiple exclamation marks (`!!`), words in ALL CAPS for emphasis, or hard-sell phrases ("BUY NOW", "LIMITED OFFER", "ORDER TODAY"). |

### Length — target 50–90 words
| Rating | Definition |
|--------|------------|
| good | 50–90 words. |
| ok | 40–49 or 91–110 words. |
| bad | Fewer than 40 or more than 110 words. |

*Counted programmatically as `len(text.split())` — **words**, not tokens. A model tokenizer
counts sub-word tokens and would give a different (wrong) number here, so the band is computed
from a plain whitespace split in Python. This makes the criterion objective. See note under
"For the judge".*

### Grounding — faithful to the source attributes (the safety criterion)
| Rating | Definition |
|--------|------------|
| good | Every factual claim is supported by the source. Generic filler is fine. |
| ok | Subjective exaggeration built on a real spec (e.g. "exceptionally powerful" for a stated 2200 W), inventing no new fact. |
| bad | At least one invented fact: a spec, connection, material, or feature not present in the source. |

### Latency — time per generation call (measured, not judged)
| Rating | Definition |
|--------|------------|
| good | ≤ 2000 ms |
| ok | 2001–5000 ms |
| bad | > 5000 ms |

*Thresholds to be re-calibrated after 2–3 warm-up calls on the actual machine. Measured by a
timer in Task 2 — **never sent to the LLM judge**, since a model cannot know how long its own
call took.*

---

## Pass / fail rules

Applied identically by hand (Task 3), after improvements (Task 4), and by the judge (Task 6).

**Cumulative pass bar:** at least **3 `good` ratings out of the 5 text/length criteria
(Fluency, Grammar, Tone, Length, Grounding — Latency excluded)** and **zero `bad` ratings
among those 5**.

Latency never counts toward the 3 required `good` ratings and never triggers a fail; it is
reported separately as a speed/cost signal only.

**Go / no-go (fails the row on its own):** if **Grounding is not `good`**, the description is
**rejected**, no matter how strong everything else is.

Consequence worth noting: because Grounding "ok" already fails the go/no-go, a description can
be fluent, grammatical, on-tone and correctly-sized and still fail on a single invented fact.
That is intended — invention is the one failure this rubric refuses to ship.

**Scope of the bar:** the pass/fail decision is over the four text criteria plus Length
(Fluency, Grammar, Tone, Length, Grounding). Latency is reported alongside as a
cost/speed signal but is not part of the go/no-go.

---

## For the judge (Task 5–6)

- The judge scores the **text criteria only**: Fluency, Grammar, Tone, Grounding.
- **Length** is computed in code (word count → band) rather than trusted to the judge, since
  it's exactly countable; the judge is handed the resulting band if needed.
- **Latency** is excluded entirely — measured programmatically.
- To score **Grounding**, the judge must receive **both** the source `description` and the
  generated text — grounding is a comparison and needs both sides.
- The judge returns, per criterion, an `explanation` **then** a `verdict` (in that order).

---

## Summary — 18 definitions at a glance

| Criterion | good | ok | bad |
|-----------|------|----|----|
| Fluency | no awkward phrasing | 1–2 clunky phrases | 3+ awkward, or must re-read |
| Grammar | zero errors | 1–2 minor errors | 3+ errors, or one that obscures meaning |
| Tone | warm & persuasive throughout | mostly on-voice, 1–2 flat spots | dry/robotic, or salesy (`!!`, ALL CAPS, "BUY NOW") |
| Length | 50–90 words | 40–49 or 91–110 | <40 or >110 |
| Grounding | all facts supported | spec-based exaggeration, no new fact | ≥1 invented fact |
| Latency | ≤2000 ms | 2001–5000 ms | >5000 ms |

**Pass =** ≥3 `good` (of the 5 text/length criteria, Latency excluded) **and** 0 `bad` among
those 5 **and** Grounding = `good`.