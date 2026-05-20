import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evaluation.judge_result import CriterionScore, JudgeResult

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

Respond ONLY in this JSON format:
{{
  "accuracy":      {{ "reasoning": "...", "score": X }},
  "relevance":     {{ "reasoning": "...", "score": X }},
  "groundedness":  {{ "reasoning": "...", "score": X }},
  "conciseness":   {{ "reasoning": "...", "score": X }},
  "overall":       {{ "reasoning": "...", "score": X }}
}}"""


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response:\n{text}")
    return json.loads(match.group())


def _parse_criterion(data: dict, key: str) -> CriterionScore:
    entry = data[key]
    return CriterionScore(reasoning=entry["reasoning"], score=int(entry["score"]))


class LLMJudge:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def judge(self, question: str, golden_answer: str, generated_answer: str) -> JudgeResult:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        data = _extract_json(response.content)
        return JudgeResult(
            accuracy=_parse_criterion(data, "accuracy"),
            relevance=_parse_criterion(data, "relevance"),
            groundedness=_parse_criterion(data, "groundedness"),
            conciseness=_parse_criterion(data, "conciseness"),
            overall=_parse_criterion(data, "overall"),
        )
