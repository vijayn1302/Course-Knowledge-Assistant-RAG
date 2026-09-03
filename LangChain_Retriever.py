import os
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 
from groq import Groq
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Check your .env file.")

print("API key loaded successfully")

model = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key = api_key,
    temperature = 0
)

#Create Documents
documents = [
    Document(
        page_content = "Langchain is a framework fro building LLM Applications."
    ),
    Document(
        page_content = "LangGraph is used to build stateful and controllable workflows."
    ),
    Document(
        page_content = "LangSmith is used to trace, debug and evaluate LLM Applications."
    )
]

#Create Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

#Create Vector Store
vector_store = InMemoryVectorStore.from_documents(
    documents,
    embeddings
)

#Create Retriever
retriever = vector_store.as_retriever(
    search_kwargs = {"k": 1}
)

# Retriever Relevant Documents
results = retriever.invoke(
    "What is LangGraph"
)

#Print Results
for doc in results:
    print(doc.page_content)