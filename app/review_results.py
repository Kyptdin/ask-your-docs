import json
import textwrap
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "evaluation" / "results"
METRICS_PATH = RESULTS_DIR / "metrics_results.jsonl"
JUDGE_PATH = RESULTS_DIR / "judge_results.jsonl"

DIVIDER = "=" * 60


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text or "(none)", width=80, initial_indent=prefix, subsequent_indent=prefix)


def _prompt_next(index: int, total: int) -> bool:
    """Returns True to continue, False to stop."""
    if index >= total - 1:
        return False
    try:
        answer = input(f"\nSee next result? ({index + 2}/{total}) [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer not in ("n", "no")


def review_metrics(records: list[dict]) -> None:
    total = len(records)
    print(f"\n{DIVIDER}")
    print(f"  METRICS RESULTS  ({total} entries)")
    print(DIVIDER)

    for i, r in enumerate(records):
        print(f"\n[{i + 1}/{total}]")
        print(f"  Category:          {r['category']}")
        print(f"  Question:")
        print(_wrap(r["question"]))
        print(f"  Generated Answer:")
        print(_wrap(r["generated_answer"]))
        print(f"  MRR:               {r['mrr']:.4f}")
        print(f"  nDCG:              {r['ndcg']:.4f}")
        print(f"  Keyword Coverage:  {r['keyword_coverage']:.4f}")

        if not _prompt_next(i, total):
            break

    print(f"\n{DIVIDER}")
    print("  End of metrics results.")
    print(DIVIDER)


def review_judge(records: list[dict]) -> None:
    total = len(records)
    criteria = ["accuracy", "relevance", "groundedness", "conciseness", "overall"]

    print(f"\n{DIVIDER}")
    print(f"  JUDGE RESULTS  ({total} entries)")
    print(DIVIDER)

    for i, r in enumerate(records):
        print(f"\n[{i + 1}/{total}]")
        print(f"  Category:      {r['category']}")
        print(f"  Question:")
        print(_wrap(r["question"]))
        print(f"  Generated Answer:")
        print(_wrap(r["generated_answer"]))
        print()
        for c in criteria:
            entry = r[c]
            print(f"  {c.capitalize():<14} score: {entry['score']}/5")
            print(_wrap(entry["reasoning"], indent=20))

        if not _prompt_next(i, total):
            break

    print(f"\n{DIVIDER}")
    print("  End of judge results.")
    print(DIVIDER)


def pick_mode() -> str:
    print("\nWhich results would you like to review?")
    print("  1) Metrics results")
    print("  2) LLM-as-a-Judge results")
    print("  3) Both (metrics first, then judge)")
    while True:
        try:
            choice = input("Choice [1/2/3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "quit"
        if choice in ("1", "2", "3"):
            return choice
        print("  Please enter 1, 2, or 3.")


def main() -> None:
    metrics = _load_jsonl(METRICS_PATH)
    judge = _load_jsonl(JUDGE_PATH)

    if not metrics and not judge:
        print("No results found. Run the evaluation first with: python run_evaluation.py")
        return

    available = []
    if metrics:
        available.append(f"metrics ({len(metrics)} entries)")
    if judge:
        available.append(f"judge ({len(judge)} entries)")
    print(f"Found: {', '.join(available)}")

    if metrics and not judge:
        review_metrics(metrics)
    elif judge and not metrics:
        review_judge(judge)
    else:
        choice = pick_mode()
        if choice == "1":
            review_metrics(metrics)
        elif choice == "2":
            review_judge(judge)
        elif choice == "3":
            review_metrics(metrics)
            review_judge(judge)


if __name__ == "__main__":
    main()
