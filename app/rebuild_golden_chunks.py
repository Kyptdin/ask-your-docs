"""
Rebuilds the golden_chunks field in golden_dataset.jsonl by matching each
question's golden_answer against the actual chunks produced from the ingested
PDFs (same splitter settings as DocumentIngester).

Run from the app/ directory:
    python rebuild_golden_chunks.py
"""

import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_APP_DIR = Path(__file__).parent
DATASET_PATH = _APP_DIR / "evaluation" / "data" / "golden_dataset.jsonl"
TEXTBOOKS = [str(_APP_DIR.parent / "textbooks" / "sol.pdf")]  # keep in ingestion order

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "or",
    "but", "for", "on", "at", "by", "with", "this", "that", "be",
    "are", "was", "were", "has", "have", "had", "not", "does", "do",
    "if", "as", "from", "which", "what", "how", "when", "where",
    "its", "their", "there", "each", "into", "than", "so", "also",
}
TOP_K = 2


def tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return tokens - STOPWORDS


def build_chunks() -> list[tuple[str, str]]:
    """Returns list of (chunk_id, page_content) in ingestion order."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    result = []
    counter = 0
    for pdf_path in TEXTBOOKS:
        docs = PyPDFLoader(pdf_path).load()
        for doc in splitter.split_documents(docs):
            result.append((f"chunk_{counter:03d}", doc.page_content))
            counter += 1
    return result


def score_chunks(
    query_tokens: set[str], chunks: list[tuple[str, str]]
) -> list[tuple[float, str]]:
    scored = []
    for chunk_id, content in chunks:
        chunk_tokens = tokenize(content)
        if not query_tokens:
            scored.append((0.0, chunk_id))
            continue
        overlap = len(query_tokens & chunk_tokens)
        score = overlap / len(query_tokens)
        scored.append((score, chunk_id))
    scored.sort(reverse=True)
    return scored


def main() -> None:
    chunks = build_chunks()
    print(f"Loaded {len(chunks)} chunks from {TEXTBOOKS}")

    questions = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    updated = 0
    for q in questions:
        if q["category"] == "unanswerable" or not q.get("golden_chunks"):
            q["golden_chunks"] = []
            continue

        answer_tokens = tokenize(q["golden_answer"])
        question_tokens = tokenize(q["question"])
        combined = answer_tokens | question_tokens

        scored = score_chunks(combined, chunks)
        top = [(s, cid) for s, cid in scored[:TOP_K] if s > 0.0]

        old = q["golden_chunks"]
        if top:
            q["golden_chunks"] = [cid for _, cid in top]
        else:
            q["golden_chunks"] = []

        if old != q["golden_chunks"]:
            updated += 1
            print(f"  [{q['id']}] {old} → {q['golden_chunks']}  (score: {scored[0][0]:.2f})")

    with open(DATASET_PATH, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"\nUpdated {updated}/{len(questions)} questions. Saved to {DATASET_PATH}")


if __name__ == "__main__":
    main()
