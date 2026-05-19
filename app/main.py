import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_ollama import ChatOllama
from pinecone import Pinecone

from ingestion.rag_pipeline import DocumentIngester
from retrieval.vector_search import DocumentRetriever
from agent.rag_agent import RAGAgent

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = PineconeEmbeddings(
    model=os.getenv("PINECONE_MODEL"),
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)
index = pc.Index(os.getenv("PINECONE_INDEX"))
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)

# ingester = DocumentIngester(vector_store)
# ingester.ingest("textbooks/sol.pdf")

retriever = DocumentRetriever(vector_store)
agent = RAGAgent(llm, retriever)

response = agent.invoke("Explain to me what question 10a is about")
print(response)
