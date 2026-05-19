from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentIngester:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

    def ingest(self, pdf_path: str) -> None:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        split_docs = self.splitter.split_documents(docs)
        self.vector_store.add_documents(documents=split_docs)
