# 📄 Ask Your Docs — PDF Q&A Tool

Upload a PDF and ask plain-language questions against it. Answers are grounded in the document with source citations.

---

## Prerequisites

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [Pinecone](https://www.pinecone.io) account (free tier works)
- **One of:** an OpenAI API key (cloud) or [Ollama](https://ollama.com) running locally

---

## Setup

**1. Clone the repo and install dependencies**

```bash
git clone <repo-url>
cd ask-your-docs
uv sync
```

**2. Create a `.env` file in the project root**

```env
# Pinecone (required)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_index_name
PINECONE_MODEL=multilingual-e5-large

# LLM — pick one of the two options below

# Option A: OpenAI (cloud)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

# Option B: Ollama (local, used only if OPENAI_API_KEY is not set)
OLLAMA_MODEL=llama3
```

**3. If using Ollama, pull your model**

```bash
ollama pull llama3
```

**4. Add your PDF**

Place the PDF you want to query inside the `textbooks/` folder.

---

## Running the App

**First run — ingest the PDF and start Q&A:**

```bash
uv run ask-your-docs --ingest
```

**Subsequent runs — PDF already indexed, skip ingestion:**

```bash
uv run ask-your-docs
```

You'll see an interactive prompt:

```
RAG pipeline ready. Type 'exit' to quit.

Ask a question: What is the purpose of the TLB?
> The TLB (Translation Lookaside Buffer) is a cache for virtual-to-physical
  address translations...
```

Type `exit` to quit.

---

## Running the Evaluation

Runs 150 questions from the golden dataset through the full pipeline and reports MRR, nDCG, and keyword coverage:

```bash
uv run run-evaluation
```

Example output:

```
========================================
Overall Results
========================================
  MRR:               0.6123
  nDCG:              0.5891
  Keyword Coverage:  0.7340
========================================
```

Results are saved to `evaluation/results/`.
