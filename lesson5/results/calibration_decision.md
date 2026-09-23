# Retriever calibration decision

**Decision (set in `lesson5/.env`):**

```
RAG_INDEX=single
RAG_MIN_RERANK_SCORE=-7.0
```

Source data: `results/calibration.json`, produced by `calibrate_threshold.py` at commit `ebe8529`.
It covers the 38 answerable lesson3 questions, the 2 unanswerable ones, and 10 off-domain queries. It uses no LLM.

## Index: `single` (`lesson3/index`, 20 candidates, rerank, top 5)

| Metric | single | multiscale | JSON path |
|---|---|---|---|
| recall@5 | 33/38 | 35/38 | `configs.<name>.summary.recall_at_5` |
| recall@1 | 28/38 | 24/38 | `configs.<name>.summary.recall_at_1` |
| MRR@5 | 0.7939 | 0.7509 | `configs.<name>.summary.mrr_at_5` |

- Single ranks the cited page first more often, so its recall@1 and MRR are higher.
- Multiscale's extra 2 hits in recall@5 are at noise level on 38 questions.
- Single also won end to end in Assignment 3: 80.0% answered correctly against 72.5% for multiscale (`lesson3/Write-up.md`, Task 6 vs Task 7).
- Single is the simpler configuration: one index instead of two.

## Threshold: -7.0

Every threshold in the open interval (-7.78, -4.47) gives the same outcome on this data:

| Value | Source in `configs.single` |
|---|---|
| lowest top score of an answerable retrieval hit: -4.47 (id 38) | `summary.threshold.lowest_answerable_hit_top_score` |
| next score below it, an answerable retrieval miss: -7.78 (id 21) | `rows[id=21].top_score` |
| highest off-domain top score: -8.13 | `summary.threshold.highest_off_domain_top_score` |

- **Why -7.0:** it sits near the permissive end of the interval. A false NO_RESULTS costs more than a weak passage, because the agent can still judge a weak passage.
- **What any value in the interval does:** all 10 off-domain queries return NO_RESULTS. Question 21, the scuba-depth question, also returns NO_RESULTS. It is answerable, but its cited page (allianz_travel_basic_fl p.17) was not in the top 5 either. No question whose evidence was retrieved gets rejected.

## What the threshold does not do

The in-domain unanswerable questions score like answerable ones: id 39 scores -0.99 and id 40 scores 0.54 (`rows[id=39|40].top_score`). The threshold therefore filters only off-domain queries. Refusing an in-domain question the corpus cannot answer is the agent's job, not the retriever's.
