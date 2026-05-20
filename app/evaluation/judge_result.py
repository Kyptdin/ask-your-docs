from dataclasses import dataclass


@dataclass
class CriterionScore:
    reasoning: str
    score: int


@dataclass
class JudgeResult:
    accuracy: CriterionScore
    relevance: CriterionScore
    groundedness: CriterionScore
    conciseness: CriterionScore
    overall: CriterionScore

    def avg_score(self) -> float:
        scores = [
            self.accuracy.score,
            self.relevance.score,
            self.groundedness.score,
            self.conciseness.score,
        ]
        return sum(scores) / len(scores)
