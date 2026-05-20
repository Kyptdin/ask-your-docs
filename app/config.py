import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone

from agent.rag_agent import RAGAgent
from ingestion.rag_pipeline import DocumentIngester
from retrieval.vector_search import DocumentRetriever

load_dotenv()

_vector_store = None


def _get_vector_store() -> PineconeVectorStore:
    global _vector_store
    if _vector_store is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        embeddings = PineconeEmbeddings(
            model=os.getenv("PINECONE_MODEL"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )
        _vector_store = PineconeVectorStore(
            embedding=embeddings,
            index=pc.Index(os.getenv("PINECONE_INDEX")),
        )
    return _vector_store


def _build_llm(provider: str | None = None):
    """
    provider=None  → auto: use OpenAI if OPENAI_API_KEY is set, else Ollama
    provider="openai" → require OpenAI
    provider="ollama" → require Ollama
    """
    resolved = provider or ("openai" if os.getenv("OPENAI_API_KEY") else "ollama")

    if resolved == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env or choose --provider ollama."
            )
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using OpenAI model: {model}")
        return ChatOpenAI(model=model, temperature=0)

    from langchain_ollama import ChatOllama
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        raise EnvironmentError("OLLAMA_MODEL is not set in your .env.")
    print(f"Using Ollama model: {model}")
    return ChatOllama(model=model, temperature=0)


def build_llm(provider: str | None = None):
    """Public wrapper around _build_llm. Safe to call per-thread."""
    return _build_llm(provider)


def build_agent(provider: str | None = None) -> RAGAgent:
    """Create a fresh RAGAgent with its own LLM client. Safe to call per-thread."""
    retriever = DocumentRetriever(_get_vector_store())
    return RAGAgent(_build_llm(provider), retriever)


def build_dependencies(provider: str | None = None) -> tuple[DocumentIngester, DocumentRetriever, RAGAgent]:
    vector_store = _get_vector_store()
    ingester = DocumentIngester(vector_store)
    retriever = DocumentRetriever(vector_store)
    agent = build_agent(provider)
    return ingester, retriever, agent
