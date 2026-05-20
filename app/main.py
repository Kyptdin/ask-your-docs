import argparse

from config import build_dependencies


def main():
    parser = argparse.ArgumentParser(description="Ask Your Docs RAG pipeline")
    parser.add_argument("--ingest", action="store_true", help="Ingest the PDF before querying")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=None,
        help="LLM backend to use (default: openai if OPENAI_API_KEY is set, else ollama)",
    )
    args = parser.parse_args()

    ingester, _, agent = build_dependencies(provider=args.provider)

    if args.ingest:
        ingester.ingest("textbooks/sol.pdf")

    print("RAG pipeline ready. Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        print(agent.invoke(query))


if __name__ == "__main__":
    main()
