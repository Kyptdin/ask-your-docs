from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from evaluation.models.judge_result import JudgeResult

_JUDGE_PROMPT = """\
You are an expert evaluator. Analyze the model's answer and score it on each criterion.

Question: {question}
Reference Answer: {golden_answer}
Model's Answer: {generated_answer}

For each criterion below, reason first, then give a score 1–5.

Criteria:
- Accuracy: Is the answer factually correct relative to the reference?
- Relevance: Does it directly address the question?
- Groundedness: Are claims supported by the source material without hallucination?
- Conciseness: Is it appropriately brief without losing key info?
- Overall: Holistic quality of the answer across all criteria."""


class LLMJudge:
    def __init__(self, llm: BaseChatModel):
        self._structured_llm = llm.with_structured_output(JudgeResult)

    def judge(self, question: str, golden_answer: str, generated_answer: str) -> JudgeResult:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
        )
        try:
            result = self._structured_llm.invoke([HumanMessage(content=prompt)])
        except ValidationError as e:
            raise ValueError(f"Judge response failed schema validation: {e}") from e
        except Exception as e:
            raise RuntimeError(f"LLM call failed during judging: {e}") from e

        if not isinstance(result, JudgeResult):
            raise TypeError(f"Expected JudgeResult, got {type(result).__name__}")

        return result
