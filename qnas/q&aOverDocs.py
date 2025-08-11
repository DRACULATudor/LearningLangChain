"""
Q&A Over Documents Tutorial
===========================
This tutorial demonstrates how to perform question-answering over CSV documents
using vector embeddings and retrieval-augmented generation (RAG).

Two approaches shown:
1. Quick setup with VectorstoreIndexCreator (Option 1)
2. Manual step-by-step approach (Option 2)
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DocArrayInMemorySearch  # Vector database for storing embeddings
from langchain_community.document_loaders import CSVLoader  # Loads CSV files into LangChain documents
from IPython.display import display, Markdown  # For pretty output formatting
from langchain.indexes import VectorstoreIndexCreator  # Simplifies vector index creation
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Google's embedding model

# Load environment variables (API keys) from .env file
_ = load_dotenv(find_dotenv())

# ==========================================
# SETUP: Initialize Models
# ==========================================

# Configure Google's embedding model - converts text to numerical vectors
# These vectors capture semantic meaning for similarity search
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Google's text embedding model
    google_api_key=os.environ['GEMINI_API_KEY']  # API key from environment
)

# Configure Google's chat model for generating responses
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Fast, efficient model for Q&A
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,  # Low temperature for consistent, factual responses
    verbose=True  # Show detailed processing steps
)

# ==========================================
# OPTION 1: Quick Setup (Recommended for beginners)
# ==========================================
# This approach uses VectorstoreIndexCreator to handle everything automatically

print("=== OPTION 1: Quick Setup ===")

# Load CSV file into LangChain document format
file = 'Products.csv'  # Your product catalog CSV file
loader = CSVLoader(file_path=file)  # Creates loader that can read CSV rows as documents

# Create a searchable vector index in one step
# This automatically: loads data → splits text → creates embeddings → stores in vector DB
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,  # Use in-memory vector store (fast, temporary)
    embedding=embeddings  # Use Google embeddings to convert text to vectors
).from_loaders([loader])  # Load from our CSV loader

# Behind the scenes process:
# CSV rows → Text chunks → Vector embeddings → Searchable index
# Each product becomes a searchable document with semantic meaning

# Ask a question about the products
query = "Tell me which product I should get, I really enjoy listening to loud music"

# Query the index - this will:
# 1. Convert query to vector embedding
# 2. Find similar product vectors (semantic search)  
# 3. Retrieve relevant product information
# 4. Generate response using LLM + retrieved context
response = index.query(query, llm=chat)

# Display the AI's recommendation
display(response)

# ==========================================
# OPTION 2: Manual Step-by-Step Approach
# ==========================================
# This shows each step explicitly for learning purposes

print("\n=== OPTION 2: Manual Step-by-Step ===")

# Step 1: Load documents from CSV
print("Step 1: Loading CSV documents...")
docs = loader.load()  # Convert CSV rows into LangChain Document objects
print(f"Loaded {len(docs)} documents from CSV")

# Step 2: Create vector database manually
print("Step 2: Creating vector database...")
db = DocArrayInMemorySearch.from_documents(
    docs,  # The documents from CSV
    embeddings  # Embedding model to convert text → vectors
)
# This creates embeddings for each CSV row and stores them for similarity search

# Step 3: Test similarity search
print("Step 3: Testing similarity search...")
docs = db.similarity_search(query)  # Find documents most similar to query
print(f"Found {len(docs)} similar documents")

# Step 4: Create retriever interface
print("Step 4: Setting up retriever...")
retriever = db.as_retriever()  # Convert vector DB to retriever interface
# Retriever provides standardized way to fetch relevant documents

# Step 5: Manual document combination (alternative approach)
print("Step 5: Manual document processing...")
# Combine all retrieved documents into one string
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

# Call LLM directly with retrieved context + new question
response = chat.call_as_llm(f"{qdocs} Question: Please list all your \
gaming products in a table in markdown and summarize each one.") 

print("Manual approach response:")
print(response)

# ==========================================
# OPTION 2B: Using RetrievalQA Chain
# ==========================================
# This uses LangChain's built-in Q&A chain for cleaner implementation

print("\n=== OPTION 2B: RetrievalQA Chain ===")

# Create a Q&A chain that automatically handles retrieval + generation
qa_stuff = RetrievalQA.from_chain_type(
    llm=chat,  # Language model for generating answers
    chain_type="stuff",  # "stuff" = put all retrieved docs into prompt context
    retriever=retriever,  # Our vector DB retriever
    verbose=True  # Show the retrieval and generation process
)
# "stuff" chain type means: retrieve docs → stuff them all into prompt → generate answer

# Ask a specific question about products
query = "Please list all your music items \
in markdown and summarize each one."

# Run the Q&A chain:
# 1. Retriever finds relevant documents about shirts with sun protection
# 2. Chain combines retrieved docs + query into a prompt
# 3. LLM generates structured response
response = qa_stuff.run(query)

# Display response as formatted markdown table
print("RetrievalQA Chain response:")
display(Markdown(response))

# ==========================================
# Summary of Approaches:
# ==========================================
# Option 1: VectorstoreIndexCreator - Quick, automatic, good for prototypes
# Option 2: Manual steps - Full control, good for learning/customization  
# Option 2B: RetrievalQA - Best of both worlds, production-ready

print("\n=== Tutorial Complete ===")
print("You've learned three ways to do Q&A over documents:")
print("1. Quick setup with VectorstoreIndexCreator")
print("2. Manual step-by-step approach") 
print("3. Production-ready RetrievalQA chain")