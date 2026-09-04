import os
from dotenv import load_dotenv 
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
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

#---------- Course Documents ----------

documents = [
    Document(
        page_content = """
        Python Course:
        The Python course covers Python fundamentals, variables, data types,
        conditional statements, loops, functions, object-oriented programming, file handling, 
        exception handling, and popular libraries such as NumPy and Pandas.""",
        metadata = {"course": "Python"}
    ),
    Document(
        page_content = """
        Machine Learning:
       The Machine Learning course covers supervised learning, unsupervised
        learning, regression, classification, clustering, feature engineering,
        model evaluation, cross-validation, and algorithms such as Linear
        Regression, Logistic Regression, Decision Trees, Random Forest,
        KNN, and SVM.""",
        metadata = {"course": "Machine Learning"}
    ),
    Document(
        page_content = """
        Generative AI Course:
        The Generative AI course covers Large Language Models, prompt
        engineering, embeddings, vector databases, Retrieval Augmented
        Generation (RAG), LangChain, LangGraph, and building AI applications.""",
        metadata = {"course": "Generative AI"}
    )
]

#---------- Embeddings ----------

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

#---------- Vector Store ----------

vector_store = InMemoryVectorStore.from_documents(documents = documents, embedding = embeddings)

#---------- Retriever ----------

retriever = vector_store.as_retriever(
    search_kwargs = {"k": 2}
)


#---------- Prompt ----------

prompt = ChatPromptTemplate.from_template(
    """ You are a helpful course assistant.

    Answer the user's question using ONLY the information
    provided in the context.

    If the answer is not available in the context,
    say that the information is not available.

    Context: {context}

    Question: {question}

    Answer: """
)

def format_docs(docs):
    return"\n\n".join(doc.page_content for doc in docs)

#----------  Parser ---------- 

parser = StrOutputParser()

#---------- Runnable Parallel ----------

parallel_chain = RunnableParallel({
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
})

#---------- Prompt -> Model -> Parser ----------

final_chain = (parallel_chain | prompt | model | parser)

#---------- User Question ----------

question = input("Ask a course-related question: ")

#---------- Generate Answer ----------

answer = final_chain.invoke(question)
print("\nAnswer:",answer)