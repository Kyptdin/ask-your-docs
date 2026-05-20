from dataclasses import dataclass


@dataclass
class EvalResult:
    mrr: float
    ndcg: float
    keyword_coverage: float
