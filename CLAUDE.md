# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Commands

```bash
uv sync                        # install dependencies
uv run ask-your-docs           # run interactive Q&A (add --ingest to ingest PDF first)
uv run ask-your-docs --ingest  # ingest textbooks/sol.pdf then start Q&A loop
uv run run-evaluation          # run full eval suite (MRR, nDCG, keyword coverage + LLM judge)
uv run python app/review_results.py   # review saved evaluation results
```

All source lives under `app/`; the package root for imports is `app/` (set via `pyproject.toml` `package-dir`). To run one-off scripts from the repo root:

```bash
uv run python -c "import sys; sys.path.insert(0, 'app'); ..."
```

## Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `PINECONE_API_KEY` | Pinecone auth |
| `PINECONE_INDEX` | Index name |
| `PINECONE_MODEL` | Embedding model (e.g. `multilingual-e5-large`) |
| `OPENAI_API_KEY` | Used by the agent LLM and by `QueryExpander` (`gpt-4.1-nano`) |
| `OPENAI_MODEL` | Agent model override (default `gpt-4o-mini`) |
| `OLLAMA_MODEL` | Used when `OPENAI_API_KEY` is absent |

## Architecture

### Request flow

```
main.py → RAGAgent.invoke(query)
            └─ LangChain agent calls retrieve_context tool
                 └─ DocumentRetriever.as_tool()
                      ├─ QueryExpander.expand(query) → [q1, q2, q3]  (gpt-4.1-nano)
                      ├─ PineconeVectorStore.similarity_search(qi, k=2)  × 3
                      └─ deduplicate by content hash → return merged chunks
```

### Key classes and their responsibilities

- **`RAGAgent`** (`agent/rag_agent.py`) — thin wrapper around a LangChain agent. Holds the system prompt and exposes `invoke(query) -> str`. Does not know about retrieval internals.
- **`DocumentRetriever`** (`retrieval/vector_search.py`) — owns the retrieval strategy. Accepts a `QueryExpander` and `k` (chunks per sub-query, default 2). Exposes both `as_tool()` (for the agent) and `retrieve()` (for the evaluation runner).
- **`QueryExpander`** (`retrieval/query_expander.py`) — calls `gpt-4.1-nano` to rewrite one query into 3 sub-queries. Always-on; no flag to disable.
- **`DocumentIngester`** (`ingestion/rag_pipeline.py`) — loads a PDF, splits into 1000-char chunks (500 overlap), assigns sequential `chunk_id` metadata (`chunk_000`, `chunk_001`, …), and upserts to Pinecone. Tracks ingested files in `ingestion/ingested_files.json` to skip re-ingestion. Chunk counter persists in `ingestion/chunk_counter.json`.
- **`config.py`** — single wiring point. `build_dependencies()` creates the vector store (singleton), `QueryExpander`, `DocumentIngester`, `DocumentRetriever`, and `RAGAgent`. `build_agent()` is a separate factory used by the evaluation runner to create per-thread agents.

### Evaluation suite

`EvaluationRunner` (`evaluation/evaluation_runner.py`) runs 150 questions from `evaluation/data/golden_dataset.jsonl` through two concurrent pipeline stages:

1. **RAG stage** (6 threads) — calls `retriever.retrieve()` + `agent.invoke()`, computes MRR, nDCG, and keyword coverage against `golden_chunks` from the dataset.
2. **Judge stage** (6 threads) — `LLMJudge` scores each generated answer on accuracy, relevance, groundedness, conciseness, and overall (1–5 scale).

Results are written to `evaluation/results/data/` (JSONL) and auto-numbered summaries to `evaluation/results/summaries/result_N.txt`.

`chunk_id` values in the golden dataset must match the `chunk_id` metadata assigned during ingestion. If the index is rebuilt, run `app/rebuild_golden_chunks.py` to regenerate the golden dataset with the new IDs.
