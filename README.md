# 🔗 LangChain Course RAG Application

A simple **Retrieval-Augmented Generation (RAG)** application built with **LangChain** that retrieves relevant course information from stored documents and uses a **Groq LLM** to generate answers.

## 🚀 Features

* 📚 Stores course-related documents
* 🔎 Retrieves relevant documents using a Retriever
* 🧠 Uses Hugging Face embeddings
* 🗄️ Uses an in-memory vector store
* ⚡ Uses `RunnableParallel`
* ➡️ Uses `RunnablePassthrough`
* 🤖 Uses Groq LLM
* 📝 Uses `ChatPromptTemplate`
* 📤 Uses `StrOutputParser`
* 🔗 Implements a complete **Prompt → Model → Parser** pipeline

## 🛠️ Technologies Used

* Python
* LangChain
* LangChain Core
* LangChain Groq
* LangChain Hugging Face
* Hugging Face Sentence Transformers
* Groq API
* InMemoryVectorStore

## 📂 Course Documents

The application contains information about:

* 🐍 **Python**

  * Python fundamentals
  * Variables and data types
  * Loops and functions
  * Object-oriented programming
  * File handling
  * NumPy and Pandas

* 🤖 **Machine Learning**

  * Supervised and unsupervised learning
  * Regression and classification
  * Clustering
  * Feature engineering
  * Model evaluation
  * Decision Trees
  * Random Forest
  * KNN
  * SVM

* ✨ **Generative AI**

  * Large Language Models
  * Prompt engineering
  * Embeddings
  * Vector databases
  * RAG
  * LangChain
  * LangGraph

## 🔄 Application Workflow

```text
User Question
      ↓
RunnableParallel
   ↙       ↘
Retriever  RunnablePassthrough
   ↓          ↓
Context    Question
   ↘        ↙
     Prompt
       ↓
     Groq LLM
       ↓
StrOutputParser
       ↓
Final Answer
```

## 🔑 Key LangChain Concepts

### RunnableParallel

Prepares the retrieved context and original question simultaneously.

```python
parallel_chain = RunnableParallel(
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
)
```

### RunnablePassthrough

Passes the original user question directly to the next step in the chain.

### Prompt → Model → Parser

The final chain follows:

```python
final_chain = (
    parallel_chain
    | prompt
    | model
    | parser
)
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/LangChain-Retriever.git
cd LangChain-Retriever
```

Install the dependencies:

```bash
pip install -U langchain langchain-core langchain-groq langchain-huggingface sentence-transformers python-dotenv
```

## 🔐 API Key Setup

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload your API key to GitHub.

Add this to `.gitignore`:

```text
.env
__pycache__/
*.pyc
```

## ▶️ Run the Application

```bash
python LangChain_Retriever.py
```

Then enter a question such as:

```text
What algorithms are covered in the Machine Learning course?
```

Example:

```text
Answer:
The Machine Learning course covers Linear Regression,
Logistic Regression, Decision Trees, Random Forest,
KNN, and SVM.
```

## 🧪 Example Questions

```text
What topics are covered in the Python course?
```

```text
Which course covers Decision Trees?
```

```text
Does the Generative AI course cover LangChain?
```

```text
What topics are included in the Machine Learning course?
```

## 📌 Project Purpose

This project demonstrates how **LangChain Runnable components** can be combined to build a basic RAG application where relevant information is retrieved from documents before generating an answer with an LLM.

## 👨‍💻 Author

**Vijay N**

If you found this project useful, feel free to ⭐ the repository.

