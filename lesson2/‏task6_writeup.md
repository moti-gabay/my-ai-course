# Task 6 — Judge Analysis & Recommendation

## 1. Judge vs human agreement (12 ground-truth rows)

| Criterion | Agreement | Direction of disagreement |
|-----------|-----------|---------------------------|
| Grammar   | 67% (8/12) | Highest agreement — near-objective criterion, as expected. |
| Tone      | 58% (7/12) | Judge is *more generous* (human `ok` -> judge `good`). |
| Grounding | 58% (7/12) | Judge is *stricter* — catches subtle distortions the human missed. |
| Fluency   | 25% (4/12) | Judge is *stricter* (human `good` -> judge `ok`), systematically. |
| **Overall** | **52%** | |

The countable criterion (Grammar) agreed most; the subjective ones (Fluency, Tone)
diverged most — the pattern the lecture predicts. The divergence is **not random
noise**: it has a consistent direction. The judge is systematically harsher on
Fluency and Grounding and more lenient on Tone, so it is a biased-but-consistent
instrument, not an erratic one.

## 2. Where we diverged, and why — the three cases

**Case A — the judge was right, the human missed it (Grounding: #8, #82).**
On the Stand Mixer (#8) the judge flagged "can control dough temperature" as
invented — the source only lists a thermal overload cut-out (a safety feature), not
temperature control. On the Mesh Wi-Fi (#82) it caught that "three nodes share a
single 5 GHz radio" misrepresents the source, which reserves a *dedicated* 5 GHz
radio for backhaul. Both are subtle factual distortions a tired human reviewer waves
through. Here the judge *adds* value — an argument **for** automation.

**Case B — the rubric was ambiguous (Fluency: #1, #36).**
On the Microwave (#1) the judge rated Fluency `ok` because "900 watts per square
inch" is a "clunky insertion"; the human rated it `good`, reading the sentence as
grammatically smooth and treating the odd content as a Grounding problem instead. On
the Blood Pressure monitor (#36) the judge called "monitoring your vital signs from
above your arm" awkward; the human didn't. The rubric's "1-2 clunky phrases = ok"
never defines whether *clunky* covers strange content or only strange syntax — so
two reasonable readers split. This is the most valuable finding: the fix is to
**tighten the rubric**, not to blame either evaluator. A revised Fluency band would
say explicitly that Fluency judges sentence structure only, and factual oddness is
scored under Grounding.

**Case C — systematic bias (Tone).**
On #44, #47, #50, #82 the human said `ok` and the judge said `good`. The judge reads
warm, competent copy as fully on-voice where the human sees flat spots. Consistent
one-directional lenience — correctable by adding sharper `good`/`ok` boundary
examples to the Tone band.

## 3. Trade-offs — human vs LLM-as-a-judge

| Dimension | Human (Task 3) | LLM judge (Task 5-6) |
|-----------|----------------|----------------------|
| Cost | ~free but my time | a few cents per 100 rows (Sonnet) |
| Speed | ~12 rows took real, tiring minutes | 100 rows in a few minutes, unattended |
| Scale | does not scale past a small sample | scales to the full 100 (and to thousands) |
| Consistency | disagrees with myself across rows and days | temperature 0 -> repeatable, including repeatably wrong |
| Accuracy | gold standard, but I missed 2 real grounding errors | matched me 52%, and caught distortions I missed |

The key tension: consistency and accuracy pull apart. The human is the accuracy
anchor but is inconsistent; the judge is highly consistent but consistently biased
(harsh on Fluency, lenient on Tone). A 52% overall agreement is not "the judge is
bad" — much of the gap is the ambiguous Fluency band (Case B), which would inflate
agreement once fixed, and part is the judge correctly overruling me (Case A).

## 4. Recommendation — for thousands of descriptions daily

It should **not** be a single choice. Recommended design:

- **Automate the bulk with the LLM judge.** It scales, it's cheap, it's repeatable,
  and it already catches grounding distortions humans miss — the failure mode that
  matters most for a retailer (an invented spec can get you sued).
- **Keep Grounding as a hard go/no-go**, scored by the judge, with the source always
  supplied. Grounding is where the judge is *stricter* than the human, which is
  exactly the safe direction for the one criterion that carries legal risk.
- **Put a human in the loop only at the escalation points**, not on every row:
  - rows the judge marks Grounding `bad` (candidate rejections) get spot-audited;
  - a small random sample per day is human-scored to track judge drift;
  - any criterion whose human/judge agreement falls below a threshold triggers a
    rubric review, not more human labelling.
- **Fix the rubric first, then re-measure.** The 25% Fluency agreement is mostly a
  rubric-definition problem (Case B). Tightening "clunky" to mean syntax-only, and
  adding boundary examples to Tone, should raise agreement materially before any
  model change is considered.

**Bottom line:** the judge is production-viable as the first-pass scorer for all
five criteria, with Grounding as an enforced gate and humans reserved for auditing
and rubric maintenance — but only after the Fluency and Tone bands are disambiguated,
since an ambiguous rubric doesn't get more accurate when you automate it: it just gets
applied at scale, silently.