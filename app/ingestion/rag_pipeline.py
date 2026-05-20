import hashlib
import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "ingested_files.json")


def _load_registry() -> dict:
    if os.path.exists(_REGISTRY_PATH):
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict) -> None:
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DocumentIngester:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

    def _chunk_id(self, doc) -> str:
        key = f"{doc.metadata.get('source', '')}::{doc.metadata.get('start_index', '')}::{doc.page_content}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def ingest(self, pdf_path: str) -> None:
        abs_path = os.path.abspath(pdf_path)
        file_hash = _file_hash(abs_path)

        registry = _load_registry()
        if registry.get(abs_path) == file_hash:
            print(f"Skipping '{pdf_path}' — already ingested.")
            return

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        split_docs = self.splitter.split_documents(docs)
        for doc in split_docs:
            doc.metadata["chunk_id"] = self._chunk_id(doc)
        ids = [doc.metadata["chunk_id"] for doc in split_docs]
        self.vector_store.add_documents(documents=split_docs, ids=ids)

        registry[abs_path] = file_hash
        _save_registry(registry)
        print(f"Ingested '{pdf_path}' ({len(split_docs)} chunks).")
