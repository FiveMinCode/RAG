import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Corrected imports for latest LangChain versions
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load LLM model from Ollama
llm = OllamaLLM(model="mistral")  # Ensure Mistral model is installed

# Load documents for retrieval
loader = TextLoader("data.txt")  # Ensure "data.txt" exists
documents = loader.load()

# Split documents into manageable chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Generate embeddings using Ollama
embedding_function = OllamaEmbeddings(model="mistral")

# Store document chunks in ChromaDB
vectorstore = Chroma.from_documents(docs, embedding_function)

# Create a retriever
retriever = vectorstore.as_retriever()

# Create the RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# User input query
query = input("Ask your question: ")
response = qa_chain.invoke({"query": query})

# Print the response
print("\nAI Response:\n", response)