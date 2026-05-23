import os
from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class _RankedChunk(BaseModel):
    index: int = Field(description="0-based index of the chunk being scored")
    score: float = Field(description="Relevance score from 1 (irrelevant) to 10 (highly relevant)")


class _RankingResult(BaseModel):
    rankings: List[_RankedChunk]


class LLMReranker:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = ChatOpenAI(
            model="gpt-4.1-nano", temperature=0, api_key=api_key
        ).with_structured_output(_RankingResult)

    def rerank(self, query: str, docs: list, top_k: int) -> list:
        if not docs:
            return []
        if len(docs) <= top_k:
            return docs

        numbered = "\n\n".join(
            f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)
        )
        prompt = (
            "You are scoring retrieved text chunks for their relevance to a user's query. "
            "For each chunk, assign a score from 1 (irrelevant) to 10 (highly relevant). "
            "Return a score for every chunk index.\n\n"
            f"Query: {query}\n\n"
            f"Chunks:\n{numbered}"
        )

        try:
            result = self.client.invoke(prompt)
            scored = {r.index: r.score for r in result.rankings}
        except Exception:
            return docs[:top_k]

        ordered = sorted(
            range(len(docs)),
            key=lambda i: scored.get(i, -1.0),
            reverse=True,
        )
        return [docs[i] for i in ordered[:top_k]]
