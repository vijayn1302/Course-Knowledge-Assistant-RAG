# LangChain-Retriever

A simple implementation of a document retriever using LangChain. This script fetches text snippets based on a user query to help an AI answer questions accurately.

# How It Works

1. SimpleRetriever: Inherits from LangChain's BaseRetriever.

2. _get_relevant_documents: The core method where you put your custom search or database lookup logic..

3. invoke(): The LangChain standard method used to pass a string query and receive a list of matching Document objects.
