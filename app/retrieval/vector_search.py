from langchain.tools import tool
from retrieval.query_expander import QueryExpander
from retrieval.reranker import LLMReranker


class DocumentRetriever:
    def __init__(
        self,
        vector_store,
        expander: QueryExpander,
        reranker: LLMReranker,
        candidate_k: int = 5,
        final_k: int = 4,
    ):
        self.vector_store = vector_store
        self.expander = expander
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.final_k = final_k

    def _retrieve_candidates(self, query: str) -> list:
        sub_queries = self.expander.expand(query)
        all_docs = []
        for sub_q in sub_queries:
            all_docs.extend(
                self.vector_store.similarity_search(sub_q, k=self.candidate_k)
            )

        seen_hashes = set()
        deduped_docs = []
        for doc in all_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                deduped_docs.append(doc)
        return deduped_docs

    def as_tool(self):
        retriever = self

        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            deduped_docs = retriever._retrieve_candidates(query)
            reranked = retriever.reranker.rerank(query, deduped_docs, retriever.final_k)

            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in reranked
            )
            return serialized, reranked

        return retrieve_context

    def retrieve(self, query: str) -> list:
        deduped_docs = self._retrieve_candidates(query)
        return self.reranker.rerank(query, deduped_docs, self.final_k)
