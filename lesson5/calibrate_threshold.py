"""
calibrate_threshold.py - compare the two lesson3 index configs and suggest a
RAG_MIN_RERANK_SCORE for each. Retrieval only, no LLM calls.

recall@5: an answerable lesson3 question is a hit when one of its cited evidence
locations (doc + page, or doc + section for the NFIP flood policy) appears in the top 5.
Multi-document questions (ids 37, 38) hit if any cited location is retrieved.

Usage:  .venv/bin/python calibrate_threshold.py
Writes: results/calibration.json
"""

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import retriever

EVAL_SET = retriever.LESSON3_DIR / "data" / "eval_set.json"
OUT = Path(__file__).resolve().parent / "results" / "calibration.json"
CONFIG_NAMES = list(retriever.CONFIGS)

OFF_DOMAIN = [
    "What is the capital of Australia?",
    "How do I bake sourdough bread at home?",
    "Explain how Kubernetes horizontal pod autoscaling works.",
    "Who won the 2018 FIFA World Cup?",
    "What is the time complexity of quicksort?",
    "Recommend a good science fiction novel.",
    "How do I translate 'good morning' into French?",
    "What is the boiling point of water at 3000 meters altitude?",
    "Write a haiku about autumn leaves.",
    "How many moons does Jupiter have?",
]


def targets(q: dict) -> List[Tuple[str, str]]:
    if q.get("evidence_section"):
        return [(retriever.NFIP_DOC, retriever.section_key(q["evidence_section"]))]
    return list(zip(q.get("evidence_docs") or [], q.get("evidence_pages") or []))


def hit_rank(hits: List[retriever.Hit], wanted: List[Tuple[str, str]]) -> Optional[int]:
    for rank, h in enumerate(hits, start=1):
        for doc, loc in wanted:
            if h.doc_name != doc:
                continue
            got = retriever.section_key(h.page) if doc == retriever.NFIP_DOC else h.page
            if got == loc:
                return rank
    return None


def run_config(config: str, questions: List[dict]) -> Dict:
    retriever.warm_up(config)
    rows, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], config=config)
        latencies.append((time.perf_counter() - t0) * 1000)
        rows.append({
            "id": q["id"],
            "category": q.get("category"),
            "answerable": q.get("answerable", True),
            "expected": targets(q),
            "rank": hit_rank(hits, targets(q)) if q.get("answerable", True) else None,
            "top_score": hits[0].score if hits else None,
            "top5": [[h.doc_name, h.page, round(h.score, 4)] for h in hits],
        })
    off = []
    for text in OFF_DOMAIN:
        hits = retriever.search(text, config=config)
        off.append({"query": text, "top_score": hits[0].score if hits else None,
                    "top_doc": f"{hits[0].doc_name} {hits[0].page}" if hits else None})
    return {"rows": rows, "off_domain": off, "median_search_ms": statistics.median(latencies),
            "nfip_placement_misses": {d: retriever.placement_misses.get(d)
                                      for d, _ in retriever.CONFIGS[config]}}


def stats(values: List[float]) -> str:
    if not values:
        return "n=0"
    return f"n={len(values)} min={min(values):.2f} median={statistics.median(values):.2f} max={max(values):.2f}"


def threshold(res: Dict) -> Dict:
    answerable = [r for r in res["rows"] if r["answerable"]]
    hit_scores = [r["top_score"] for r in answerable if r["rank"] is not None]
    off_scores = [o["top_score"] for o in res["off_domain"]]
    lo, hi = min(hit_scores), max(off_scores)
    out = {"lowest_answerable_hit_top_score": lo, "highest_off_domain_top_score": hi,
           "separable": lo > hi, "suggested": (lo + hi) / 2 if lo > hi else None}
    t = out["suggested"]
    if t is not None:
        out["answerable_rejected_at_suggested"] = sum(r["top_score"] < t for r in answerable)
        out["answerable_hits_rejected_at_suggested"] = sum(
            r["top_score"] < t for r in answerable if r["rank"] is not None)
        out["unanswerable_rejected_at_suggested"] = sum(
            r["top_score"] < t for r in res["rows"] if not r["answerable"])
        out["off_domain_accepted_at_suggested"] = sum(s >= t for s in off_scores)
    return out


def fmt_rank(row: Dict) -> str:
    if not row["answerable"]:
        return "  n/a"
    return f"   {row['rank']}" if row["rank"] else "   - "


def main() -> None:
    questions = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    results = {c: run_config(c, questions) for c in CONFIG_NAMES}
    answerable = [q for q in questions if q.get("answerable", True)]

    print(f"Eval set: {EVAL_SET}  ({len(questions)} questions, {len(answerable)} answerable)")
    print("Rank = position of the first cited doc+page (or NFIP section) in the top 5; '-' = miss.\n")
    header = f"{'id':>3}  {'category':<12} {'expected (doc, page/section)':<58}"
    for c in CONFIG_NAMES:
        header += f" | {c:>10} rank  top"
    print(header)
    print("-" * len(header))
    by_id = {c: {r["id"]: r for r in results[c]["rows"]} for c in CONFIG_NAMES}
    for q in questions:
        exp = "; ".join(f"{d.replace('.pdf', '')} {p}" for d, p in targets(q)) or "(none)"
        line = f"{q['id']:>3}  {q.get('category', ''):<12} {exp[:58]:<58}"
        for c in CONFIG_NAMES:
            r = by_id[c][q["id"]]
            line += f" | {fmt_rank(r):>15} {r['top_score']:6.2f}"
        print(line)

    print("\nMisses: what the top 3 were instead")
    for c in CONFIG_NAMES:
        for r in results[c]["rows"]:
            if r["answerable"] and r["rank"] is None:
                top3 = ", ".join(f"{d.replace('.pdf', '')} {p} ({s:.2f})" for d, p, s in r["top5"][:3])
                print(f"  [{c}] id {r['id']}: {top3}")

    print("\nOff-domain queries (top score, top doc)")
    for i, text in enumerate(OFF_DOMAIN):
        line = f"  {text[:52]:<52}"
        for c in CONFIG_NAMES:
            o = results[c]["off_domain"][i]
            line += f" | {c}: {o['top_score']:6.2f} {o['top_doc'][:34]:<34}"
        print(line)

    print("\nSummary")
    for c in CONFIG_NAMES:
        res = results[c]
        rows = res["rows"]
        ans = [r for r in rows if r["answerable"]]
        hits = [r for r in ans if r["rank"] is not None]
        ranks = [r["rank"] for r in hits]
        res["summary"] = {
            "recall_at_5": f"{len(hits)}/{len(ans)}",
            "recall_at_1": f"{sum(k == 1 for k in ranks)}/{len(ans)}",
            "mrr_at_5": round(sum(1 / k for k in ranks) / len(ans), 4),
            "threshold": threshold(res),
        }
        th = res["summary"]["threshold"]
        print(f"\n  [{c}]  indexes: {[d for d, _ in retriever.CONFIGS[c]]}")
        print(f"    recall@5            {res['summary']['recall_at_5']}")
        print(f"    recall@1            {res['summary']['recall_at_1']}")
        print(f"    MRR@5               {res['summary']['mrr_at_5']}")
        print(f"    median search ms    {res['median_search_ms']:.0f}")
        print(f"    NFIP chunks unplaced {res['nfip_placement_misses']}")
        print(f"    top score, answerable hit   {stats([r['top_score'] for r in hits])}")
        print(f"    top score, answerable miss  {stats([r['top_score'] for r in ans if r['rank'] is None])}")
        print(f"    top score, unanswerable     {stats([r['top_score'] for r in rows if not r['answerable']])}")
        print(f"    top score, off-domain       {stats([o['top_score'] for o in res['off_domain']])}")
        print(f"    lowest answerable-hit top score  {th['lowest_answerable_hit_top_score']:.2f}")
        print(f"    highest off-domain top score     {th['highest_off_domain_top_score']:.2f}")
        if th["separable"]:
            print(f"    suggested RAG_MIN_RERANK_SCORE   {th['suggested']:.2f} (midpoint)")
            print(f"      at that threshold: answerable rejected {th['answerable_rejected_at_suggested']}/{len(ans)}"
                  f" (of which retrieval hits {th['answerable_hits_rejected_at_suggested']}),"
                  f" unanswerable rejected {th['unanswerable_rejected_at_suggested']}/{len(rows) - len(ans)},"
                  f" off-domain accepted {th['off_domain_accepted_at_suggested']}/{len(OFF_DOMAIN)}")
        else:
            print("    suggested RAG_MIN_RERANK_SCORE   none: answerable-hit and off-domain scores overlap")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"eval_set": str(EVAL_SET), "off_domain_queries": OFF_DOMAIN,
                               "configs": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
