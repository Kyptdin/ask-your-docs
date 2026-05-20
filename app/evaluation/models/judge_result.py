from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


class JudgeResult(BaseModel):
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
