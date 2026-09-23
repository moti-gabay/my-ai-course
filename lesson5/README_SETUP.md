# Setup

Everything in `lesson5/` runs from its own virtual environment built from the pinned `requirements.txt`. It needs Python 3.10+.

```bash
cd lesson5
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# 1. CPU-only torch first. The embeddings (bge-small) and the CrossEncoder reranker
#    run on CPU, so the default CUDA wheel is about 2 GB of dead weight (notably in WSL).
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else. The torch pin is already satisfied by 2.13.0+cpu.
.venv/bin/pip install -r requirements.txt
```

Put API keys in `lesson5/.env` (gitignored):

```
ANTHROPIC_API_KEY=...
```

The retriever reads the Assignment 3 FAISS indexes from `../lesson3/`, so run from inside `lesson5/` with the repo checked out in full.

Dashboard:

```bash
.venv/bin/streamlit run dashboard/app.py
```
