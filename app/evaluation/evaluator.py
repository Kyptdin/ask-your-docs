import math
import re

from evaluation.eval_result import EvalResult


class Evaluator:
    def compute_mrr(self, retrieved_chunk_ids: list[str], golden_chunks: list[str]) -> float:
        if not golden_chunks:
            return 1.0
        golden_set = set(golden_chunks)
        for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
            if chunk_id in golden_set:
                return 1.0 / rank
        return 0.0

    def compute_ndcg(self, retrieved_chunk_ids: list[str], golden_chunks: list[str]) -> float:
        if not golden_chunks:
            return 1.0
        golden_set = set(golden_chunks)

        def dcg(ids: list[str]) -> float:
            return sum(
                1.0 / math.log2(rank + 1)
                for rank, chunk_id in enumerate(ids, start=1)
                if chunk_id in golden_set
            )

        ideal_ids = list(golden_set) + [""] * max(0, len(retrieved_chunk_ids) - len(golden_set))
        idcg = dcg(ideal_ids[: len(retrieved_chunk_ids)])
        if idcg == 0:
            return 0.0
        return dcg(retrieved_chunk_ids) / idcg

    def compute_keyword_coverage(self, golden_answer: str, generated_answer: str) -> float:
        keywords = set(re.findall(r"\b\w+\b", golden_answer.lower()))
        stopwords = {
            "the", "a", "an", "is", "it", "in", "of", "to", "and", "or",
            "but", "for", "on", "at", "by", "with", "this", "that", "be",
            "are", "was", "were", "has", "have", "had", "not", "does", "do",
        }
        keywords -= stopwords
        if not keywords:
            return 1.0
        generated_tokens = set(re.findall(r"\b\w+\b", generated_answer.lower()))
        return len(keywords & generated_tokens) / len(keywords)

    def evaluate(
        self,
        golden_answer: str,
        golden_chunks: list[str],
        generated_answer: str,
        retrieved_chunk_ids: list[str],
    ) -> EvalResult:
        return EvalResult(
            mrr=self.compute_mrr(retrieved_chunk_ids, golden_chunks),
            ndcg=self.compute_ndcg(retrieved_chunk_ids, golden_chunks),
            keyword_coverage=self.compute_keyword_coverage(golden_answer, generated_answer),
        )
