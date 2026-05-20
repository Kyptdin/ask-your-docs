from dataclasses import dataclass, field

from evaluation.models.eval_result import EvalResult
from evaluation.evaluator import Evaluator
from evaluation.models.judge_result import JudgeResult


@dataclass
class TestQuestion:
    id: str
    question: str
    golden_answer: str
    golden_chunks: list[str]
    answer_type: str
    difficulty: str
    category: str
    generated_answer: str | None = field(default=None, init=False)
    metrics: EvalResult | None = field(default=None, init=False)
    judge_result: JudgeResult | None = field(default=None, init=False)

    def compute_metrics(
        self,
        generated_answer: str,
        retrieved_chunk_ids: list[str],
        evaluator: Evaluator | None = None,
    ) -> EvalResult:
        if evaluator is None:
            evaluator = Evaluator()
        self.metrics = evaluator.evaluate(
            golden_answer=self.golden_answer,
            golden_chunks=self.golden_chunks,
            generated_answer=generated_answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
        return self.metrics

    @classmethod
    def from_dict(cls, data: dict) -> "TestQuestion":
        return cls(
            id=data["id"],
            question=data["question"],
            golden_answer=data["golden_answer"],
            golden_chunks=data["golden_chunks"],
            answer_type=data["answer_type"],
            difficulty=data["difficulty"],
            category=data["category"],
        )
