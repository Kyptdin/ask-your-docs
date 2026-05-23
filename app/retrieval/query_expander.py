import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class QueryExpander:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=api_key)

    def expand(self, query: str) -> list[str]:
        prompt = (
            f"Rewrite the following query into exactly 3 distinct search sub-queries that together "
            f"cover the full breadth of the question. Output only the 3 queries, one per line, "
            f"with no numbering, bullets, or extra text.\n\nQuery: {query}"
        )

        message = HumanMessage(content=prompt)
        response = self.client.invoke([message])

        lines = response.content.strip().split('\n')
        queries = [line.strip() for line in lines if line.strip()]

        return queries[:3]
