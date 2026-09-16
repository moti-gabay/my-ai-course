# RAG Pipeline Evaluation & Optimization Write-up

## Task 1: Baseline Evaluation (Without RAG)
* **Setup:** Direct evaluation of the LLM (Claude Haiku) without providing external context or policy documents.
* **Results:** 
  * Answered correctly: **15.0%** (6/40)
  * Refused: **70.0%** (28/40)
  * Hallucinated: **15.0%** (6/40)
* **Key Findings:** Without external context, the base model lacks granular, domain-specific insurance policy knowledge. Due to strict system prompts instructing it not to speculate, it correctly refused the vast majority of domain-specific questions, while occasionally hallucinating policy terms when attempting to answer.

## Task 3: Initial RAG Implementation & Inspection
* **Setup:** Naive dense retrieval using BGE embeddings (`BAAI/bge-small-en-v1.5`) and FAISS vector store with `top_k=3`.
* **Results:** 
  * Answered correctly: **52.5%** (21/40)
  * Refused: **40.0%** (16/40)
  * Hallucinated: **7.5%** (3/40)
* **Key Findings:** Introducing RAG produced an immediate +37.5% jump in accuracy. Inspection of retrieved chunks revealed two main limitations: arbitrary text splitting cut off exclusion clauses mid-sentence, and `top_k=3` was insufficient for multi-clause or comparative questions.

## Task 5: Chunking & Context Optimization
* **Setup:** Updated text splitting in `build_index.py` (`chunk_size=1000`, `chunk_overlap=200`) and increased retrieval context window to `top_k=6`.
* **Results:** 
  * Answered correctly: **60.0%** (24/40)
  * Refused: **27.5%** (11/40)
  * Hallucinated: **12.5%** (5/40)
* **Key Findings:** Overlapping chunks prevented context truncation and reduced refusals down to 27.5%. However, feeding more unranked chunks into the prompt introduced background noise, causing a slight bump in hallucination rate (12.5%).

## Task 6: Advanced Two-Stage Retrieval (CrossEncoder Reranking)
* **Setup:** Two-stage pipeline: broad retrieval of 20 candidate chunks from FAISS (`initial_top_k=20`), followed by semantic reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`, filtering down to the top 5 most relevant chunks for the LLM.
* **Results:** 
  * Answered correctly: **80.0%** (32/40)
  * Refused: **15.0%** (6/40)
  * Hallucinated: **5.0%** (2/40)
* **Key Findings:** Reranking successfully bridged lexical gaps (e.g., mapping user terms like "hotel" to policy terms like "Additional Living Expenses") and resolved multi-document competition (e.g., retrieving policies for both Embrace and Nationwide simultaneously).

---

### Comparison Summary

| Metric | Task 1 (Baseline) | Task 3 (Basic RAG) | Task 5 (Optimized Chunking) | Task 6 (CrossEncoder Reranking) |
|---|---|---|---|---|
| **Answered Correctly** | 15.0% | 52.5% | 60.0% | **80.0%** |
| **Refused** | 70.0% | 40.0% | 27.5% | **15.0%** |
| **Hallucinated** | 15.0% | 7.5% | 12.5% | **5.0%** |
| **Avg. Latency** | ~0.8s | ~1.2s | ~1.5s | **~2.8s** |

---
## Task 7: Multi-Scale Chunking Analysis (Bonus)

### Setup & Hypothesis
* **Hypothesis:** Combining multiple vector indices with varying chunk granularities—specifically a small chunk index (`chunk_size=300`) for precise factual lookups and a large chunk index (`chunk_size=1200`) for broader context—merged via a CrossEncoder reranker, will improve search precision across heterogeneous query types.
* **Pipeline Configuration:** 
  * Two FAISS indices (`faiss_index_small` & `faiss_index_large`).
  * Retrieved top-10 candidate chunks from each index (20 candidates total).
  * Merged, deduplicated, and reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2` down to top-5 chunks.

### Empirical Results

| Metric | Single Index Rerank (Task 6) | Multi-Scale Chunking (Task 7) | Delta |
|---|---|---|---|
| **Answered Correctly** | **80.0%** (32/40) | **72.5%** (29/40) | -7.5% |
| **Refused** | **15.0%** (6/40) | **17.5%** (7/40) | +2.5% |
| **Hallucinated** | **5.0%** (2/40) | **10.0%** (4/40) | +5.0% |
| **Avg. Latency** | **~2.8s** | **~2.85s** | +0.05s |

### Key Findings & Per-Question Evidence

1. **Granularity Trade-off:** Contrary to the initial hypothesis, querying multiple chunk sizes simultaneously introduced overlapping contextual noise into the final prompt context window.
2. **Short vs. Broad Query Dynamics:**
   * **Short factual lookups** (e.g., specific monetary thresholds, liability limits) benefited from `chunk_size=300` as the retrieved text contained zero surrounding distraction.
   * **Broad explanatory questions** (e.g., exclusions across multiple policy conditions) failed under `chunk_size=300` due to cut-off boundaries, preferring the `chunk_size=1200` index.
3. **Conclusion:** A uniform `chunk_size=1000` with `overlap=200` coupled with a CrossEncoder reranker (Task 6) remains the superior baseline. Multi-scale chunking increases the risk of returning duplicate/fragmented clauses unless accompanied by strict metadata hierarchy filtering (e.g., Parent-Document Retrieval).
---

## מה הייתי מתקן מחר בבוקר (ומה המדד שכיוון לזה)

המדד המרכזי שמכוון את הצעד הבא הוא **שיעור הסירובים שנשאר על 15.0% (6/40)**. ניתוח השאלות שסורבו ב-`rag_results.json` מראה כי הסיבה העיקרית לכישלון היא **פער במילות מפתח מדויקות (Keyword Matching)** וציטוטים של מספרי סעיפים/חוקים ספציפיים שחיפוש וקטורי דנס (Dense Retrieval) לא תמיד מעלה ל-20 התוצאות הראשונות. מחר בבוקר הייתי מיישם **Hybrid Search (שילוב BM25 יחד עם FAISS)** כדי לקלוט מונחים טכניים ומזהים מדויקים, לצד מנגנון **Metadata Filtering** המבצע סינון קשיח לפי שם חברת הביטוח במידה והיא מצוינת במפורש בשאלה.