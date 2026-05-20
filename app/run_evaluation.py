import argparse

from config import build_dependencies, build_agent, build_llm
from evaluation.evaluation_runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=None,
        help="LLM backend to use (default: openai if OPENAI_API_KEY is set, else ollama)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-a-Judge stage",
    )
    args = parser.parse_args()

    _, retriever, _ = build_dependencies(provider=args.provider)

    llm_factory = None if args.no_judge else (lambda: build_llm(args.provider))

    runner = EvaluationRunner(
        retriever,
        agent_factory=lambda: build_agent(args.provider),
        llm_factory=llm_factory,
    )
    runner.run()


if __name__ == "__main__":
    main()
