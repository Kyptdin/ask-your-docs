from langchain.tools import tool


class DocumentRetriever:
    def __init__(self, vector_store, k: int = 6):
        self.vector_store = vector_store
        self.k = k

    def as_tool(self):
        vector_store = self.vector_store
        k = self.k

        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            retrieved_docs = vector_store.similarity_search(query, k=k)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        return retrieve_context

    def retrieve(self, query: str) -> list:
        return self.vector_store.similarity_search(query, k=self.k)
