import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "ingested_files.json")
_COUNTER_PATH = os.path.join(os.path.dirname(__file__), "chunk_counter.json")


def _load_registry() -> dict:
    if os.path.exists(_REGISTRY_PATH):
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict) -> None:
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _load_counter() -> int:
    if os.path.exists(_COUNTER_PATH):
        with open(_COUNTER_PATH) as f:
            return json.load(f).get("next", 0)
    return 0


def _save_counter(value: int) -> None:
    with open(_COUNTER_PATH, "w") as f:
        json.dump({"next": value}, f)


class DocumentIngester:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            add_start_index=True,
        )

    def ingest(self, pdf_path: str) -> None:
        abs_path = os.path.abspath(pdf_path)

        registry = _load_registry()
        if abs_path in registry:
            print(f"Skipping '{pdf_path}' — already ingested.")
            return

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        split_docs = self.splitter.split_documents(docs)

        start = _load_counter()
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = f"chunk_{start + i:03d}"
        _save_counter(start + len(split_docs))

        ids = [doc.metadata["chunk_id"] for doc in split_docs]
        self.vector_store.add_documents(documents=split_docs, ids=ids)

        registry[abs_path] = True
        _save_registry(registry)
        print(f"Ingested '{pdf_path}' ({len(split_docs)} chunks, ids chunk_{start:03d}–chunk_{start + len(split_docs) - 1:03d}).")
