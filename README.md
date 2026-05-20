# Ask Your Docs — PDF Q&A Tool

Upload PDFs and ask plain-language questions against them.
Uses Pinecone for vector search, Ollama for local LLM inference, and a RAG agent to generate answers grounded in the uploaded documents.

---

## Installation

**Prerequisites**
- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com) running locally

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd ask-your-docs
uv sync
```

**2. Configure environment variables**

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_index_name
PINECONE_MODEL=multilingual-e5-large
OLLAMA_MODEL=llama3
```

**3. Pull your Ollama model**

```bash
ollama pull llama3
```

---

## Running the App

Place your PDF inside the `textbooks/` folder, then run:

```bash
uv run ask-your-docs
```

This ingests the PDF into Pinecone and starts an interactive question loop:

```
RAG pipeline ready. Type 'exit' to quit.

Ask a question: What is the purpose of the TLB?
> The TLB (Translation Lookaside Buffer) is a cache for virtual-to-physical address translations...
```

Type `exit` to quit.

---

## Running the Evaluation

The evaluation suite runs 150 questions from the golden dataset through the full RAG pipeline and reports MRR, nDCG, and keyword coverage:

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

MRR by Category
----------------------------------------
  multi_hop            0.5500  (n=42)
  spanning             0.6800  (n=38)
  unanswerable         1.0000  (n=70)
========================================
```
