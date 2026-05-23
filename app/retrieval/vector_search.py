from langchain.tools import tool
from retrieval.query_expander import QueryExpander


class DocumentRetriever:
    def __init__(self, vector_store, expander: QueryExpander, k: int = 2):
        self.vector_store = vector_store
        self.expander = expander
        self.k = k

    def as_tool(self):
        vector_store = self.vector_store
        expander = self.expander
        k = self.k

        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            sub_queries = expander.expand(query)
            all_docs = []
            for sub_q in sub_queries:
                all_docs.extend(vector_store.similarity_search(sub_q, k=k))

            seen_hashes = set()
            deduped_docs = []
            for doc in all_docs:
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    deduped_docs.append(doc)

            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in deduped_docs
            )
            return serialized, deduped_docs

        return retrieve_context

    def retrieve(self, query: str) -> list:
        sub_queries = self.expander.expand(query)
        all_docs = []
        for sub_q in sub_queries:
            all_docs.extend(self.vector_store.similarity_search(sub_q, k=self.k))

        seen_hashes = set()
        deduped_docs = []
        for doc in all_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                deduped_docs.append(doc)

        return deduped_docs
