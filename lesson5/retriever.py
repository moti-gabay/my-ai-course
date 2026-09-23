"""
retriever.py - the Assignment 3 retriever, reused as the agents' search layer.

Two index configurations from lesson3, selected by RAG_INDEX (or the `config` arg):
  single      lesson3/index (chunk 1000/200), 20 FAISS candidates -> CrossEncoder -> top 5
  multiscale  lesson3/faiss_index_small (300/50) + faiss_index_large (1200/200),
              10 candidates each -> dedup -> CrossEncoder -> top 5
There is deliberately no default: the choice is made from calibrate_threshold.py output.

The NFIP flood policy is markdown with no pages, so its chunks are labelled with the
section they fall in ("§ 61.5", "Appendix A(1), V. Exclusions"). That label plays the
role of the page number everywhere (search results, read_page, recall scoring).
"""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))  # RAG_INDEX / RAG_MIN_RERANK_SCORE; never overrides the shell

LESSON3_DIR = Path(__file__).resolve().parents[1] / "lesson3"
CORPUS_DIR = LESSON3_DIR / "corpus"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      # must match lesson3/build_index.py
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NFIP_DOC = "nfip_flood_policy_44cfr61.md"
EXCLUDED_DOCS = {"README.md"}
FETCH_MULTIPLIER = 10  # FAISS pool scanned per candidate so filters still return a full k

# config name -> [(index dir under lesson3, candidates from that index)]
CONFIGS: Dict[str, List[Tuple[str, int]]] = {
    "single": [("index", 20)],
    "multiscale": [("faiss_index_small", 10), ("faiss_index_large", 10)],
}

POLICY_ALIASES: Dict[str, str] = {
    "auto": "allstate_auto_policy.pdf",
    "embrace": "embrace_pet_policy.pdf",
    "nationwide": "nationwide_pet_medical_plan.pdf",
    "travel_fl": "allianz_travel_basic_fl.pdf",
    "travel_uk": "allianz_travel_policy_wording.pdf",
    "health": "mhbp_health_evidence_of_coverage.pdf",
    "flood": NFIP_DOC,
}


@dataclass(frozen=True)
class Hit:
    doc_name: str
    page: str      # "4" for PDFs, a section label for the NFIP markdown
    score: float   # CrossEncoder rerank score (higher = more relevant)
    text: str


def resolve_config(config: Optional[str] = None) -> str:
    name = config or os.environ.get("RAG_INDEX", "")
    if name not in CONFIGS:
        raise ValueError(f"RAG_INDEX must be one of {sorted(CONFIGS)}, got {name!r}")
    return name


def resolve_doc(doc: str) -> str:
    """Accept a policy alias or a corpus file name."""
    name = POLICY_ALIASES.get(doc.strip().lower(), doc.strip())
    if name not in set(POLICY_ALIASES.values()):
        raise ValueError(f"unknown policy {doc!r}; use one of {sorted(POLICY_ALIASES)}")
    return name


# ---------------------------------------------------------------------------
# NFIP sections
# ---------------------------------------------------------------------------

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


@lru_cache(maxsize=None)
def _nfip_text() -> str:
    return (CORPUS_DIR / NFIP_DOC).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _nfip_sections() -> List[Tuple[int, str]]:
    """(start offset, label) for every section, in file order.

    '#### ' headings give "§ 61.5" and "Appendix A(1)". Inside an appendix, the
    policy parts ("V. Exclusions") are plain lines; only the next expected roman
    numeral counts, which skips look-alikes such as "I. No Benefit to Bailee".
    """
    sections: List[Tuple[int, str]] = []
    appendix, expected, offset = None, 0, 0
    for line in _nfip_text().splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            m = re.match(r"(§\s*61\.\d+)", heading)
            if m:
                appendix, label = None, m.group(1)
            elif heading.startswith("Appendix"):
                appendix, expected = heading.replace(" to Part 61", ""), 0
                label = appendix
            else:
                appendix, label = None, heading
            sections.append((offset, label))
        elif appendix and expected < len(_ROMAN) and stripped.startswith(_ROMAN[expected] + ". "):
            sections.append((offset, f"{appendix}, {stripped}"))
            expected += 1
        offset += len(line)
    return sections


def _section_at(offset: int) -> str:
    label = "N/A"
    for start, name in _nfip_sections():
        if start > offset:
            break
        label = name
    return label


def section_key(label: str) -> str:
    """Canonical key for matching NFIP sections: '§ 61.5' or 'Appendix A(1)|V'."""
    m = re.search(r"61\.(\d+)", label)
    if m and "Appendix" not in label:
        return f"§ 61.{m.group(1)}"
    m = re.search(r"Appendix\s+A\((\d)\)(?:\s*,\s*([IVX]+)\.)?", label)
    if m:
        return f"Appendix A({m.group(1)})" + (f"|{m.group(2)}" if m.group(2) else "")
    return label.strip()


def read_section(label: str) -> str:
    key = section_key(label)
    sections = _nfip_sections()
    for i, (start, name) in enumerate(sections):
        if section_key(name) == key:
            end = sections[i + 1][0] if i + 1 < len(sections) else len(_nfip_text())
            return _nfip_text()[start:end].strip()
    raise ValueError(f"section {label!r} not found in {NFIP_DOC}")


# ---------------------------------------------------------------------------
# Models and indexes (loaded once, lazily)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=None)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


placement_misses: Dict[str, int] = {}


@lru_cache(maxsize=None)
def _store(index_dir: str):
    """Load a lesson3 FAISS index and label its NFIP chunks with their section.

    The docstore keeps chunks in insertion order, which is file order, so each NFIP
    chunk is located by searching forward from the previous chunk's start. That keeps
    the verbatim-repeated clauses in Appendix A(1)/A(2)/A(3) apart.
    """
    from langchain_community.vectorstores import FAISS
    store = FAISS.load_local(str(LESSON3_DIR / index_dir), _embeddings(),
                             allow_dangerous_deserialization=True)
    text, cursor, misses = _nfip_text(), 0, 0
    for pos in sorted(store.index_to_docstore_id):
        doc = store.docstore.search(store.index_to_docstore_id[pos])
        if doc.metadata.get("doc_name") != NFIP_DOC:
            continue
        found = text.find(doc.page_content, cursor)
        if found < 0:
            found = text.find(doc.page_content[:120], cursor)
        if found < 0:
            misses += 1
            doc.metadata["page"] = "N/A"
            continue
        doc.metadata["page"] = _section_at(found)
        cursor = found + 1
    placement_misses[index_dir] = misses
    return store


def warm_up(config: Optional[str] = None) -> None:
    for index_dir, _ in CONFIGS[resolve_config(config)]:
        _store(index_dir)
    _reranker()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, policy: Optional[str] = None, config: Optional[str] = None,
           top_n: int = 5) -> List[Hit]:
    """FAISS candidates -> CrossEncoder rerank -> top_n hits, best first."""
    config = resolve_config(config)
    doc_filter = resolve_doc(policy) if policy else None

    def keep(md: dict) -> bool:
        name = md.get("doc_name")
        return name not in EXCLUDED_DOCS and (doc_filter is None or name == doc_filter)

    candidates, seen = [], set()
    for index_dir, k in CONFIGS[config]:
        for doc in _store(index_dir).similarity_search(
                query, k=k, filter=keep, fetch_k=k * FETCH_MULTIPLIER):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                candidates.append(doc)
    if not candidates:
        return []

    scores = _reranker().predict([[query, d.page_content] for d in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [Hit(d.metadata["doc_name"], str(d.metadata.get("page", "N/A")), float(s), d.page_content)
            for d, s in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Page reader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _pdf(doc_name: str):
    from pypdf import PdfReader
    return PdfReader(str(CORPUS_DIR / doc_name))


def read_page(doc: str, page: str) -> str:
    """Full text of one PDF page (1-based, as in lesson3 metadata) or one NFIP section."""
    doc_name = resolve_doc(doc)
    if doc_name == NFIP_DOC:
        return read_section(page)
    m = re.search(r"\d+", str(page))
    if not m:
        raise ValueError(f"page must be a number for {doc_name}, got {page!r}")
    number, reader = int(m.group()), _pdf(doc_name)
    if not 1 <= number <= len(reader.pages):
        raise ValueError(f"{doc_name} has pages 1-{len(reader.pages)}, got {number}")
    return reader.pages[number - 1].extract_text() or ""
