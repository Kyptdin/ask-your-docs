# Graph Report - /home/isaac/ask-your-docs  (2026-05-23)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 87 nodes · 112 edges · 18 communities (12 shown, 6 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b6866cb5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `build_agent()` - 8 edges
2. `build_dependencies()` - 7 edges
3. `DocumentRetriever` - 7 edges
4. `LLMJudge` - 6 edges
5. `TestQuestion` - 6 edges
6. `main()` - 5 edges
7. `RAGAgent` - 5 edges
8. `JudgeResult` - 5 edges
9. `main()` - 4 edges
10. `_prompt_next()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `build_dependencies()`  [INFERRED]
  app/main.py → app/config.py
- `build_agent()` --calls--> `DocumentRetriever`  [INFERRED]
  app/config.py → app/retrieval/vector_search.py
- `build_agent()` --calls--> `RAGAgent`  [INFERRED]
  app/config.py → app/agent/rag_agent.py
- `build_dependencies()` --calls--> `DocumentIngester`  [INFERRED]
  app/config.py → app/ingestion/rag_pipeline.py
- `build_dependencies()` --calls--> `DocumentRetriever`  [INFERRED]
  app/config.py → app/retrieval/vector_search.py

## Communities (18 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.24
Nodes (5): BaseModel, EvalResult, CriterionScore, JudgeResult, TestQuestion

### Community 1 - "Community 1"
Cohesion: 0.24
Nodes (9): build_agent(), build_dependencies(), _build_llm(), _get_vector_store(), provider=None  → auto: use OpenAI if OPENAI_API_KEY is set, else Ollama     prov, Public wrapper around _build_llm. Safe to call per-thread., Create a fresh RAGAgent with its own LLM client. Safe to call per-thread., main() (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.44
Nodes (8): _load_jsonl(), main(), pick_mode(), _prompt_next(), Returns True to continue, False to stop., review_judge(), review_metrics(), _wrap()

### Community 5 - "Community 5"
Cohesion: 0.39
Nodes (5): DocumentIngester, _load_counter(), _load_registry(), _save_counter(), _save_registry()

### Community 6 - "Community 6"
Cohesion: 0.48
Nodes (6): build_chunks(), main(), Rebuilds the golden_chunks field in golden_dataset.jsonl by matching each questi, Returns list of (chunk_id, page_content) in ingestion order., score_chunks(), tokenize()

### Community 7 - "Community 7"
Cohesion: 0.50
Nodes (4): Ask Your Docs — PDF Q&A Tool, Ollama, Pinecone, RAG agent

## Knowledge Gaps
- **7 isolated node(s):** `next`, `allow`, `Pinecone`, `Ollama`, `RAG agent` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 1` to `Community 2`?**
  _High betweenness centrality (0.199) - this node is a cross-community bridge._
- **Why does `build_dependencies()` connect `Community 1` to `Community 3`, `Community 5`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Why does `build_agent()` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `build_agent()` (e.g. with `DocumentRetriever` and `RAGAgent`) actually correct?**
  _`build_agent()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `build_dependencies()` (e.g. with `main()` and `DocumentIngester`) actually correct?**
  _`build_dependencies()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `DocumentRetriever` (e.g. with `build_agent()` and `build_dependencies()`) actually correct?**
  _`DocumentRetriever` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `LLMJudge` (e.g. with `evaluation_runner.py` and `.run()`) actually correct?**
  _`LLMJudge` has 3 INFERRED edges - model-reasoned connections that need verification._