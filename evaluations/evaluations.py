"""
Q&A System Evaluation Tutorial
==============================
This demonstrates how to evaluate a Q&A system using LLMs themselves as evaluators.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAGenerateChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import CSVLoader
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator
import pandas as pd
from langchain.evaluation.qa import QAEvalChain
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables (API keys) from .env file
_=load_dotenv(find_dotenv())

# ==========================================
# SETUP: Initialize AI Models
# ==========================================

# Embedding model: Converts text to numerical vectors for similarity search
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ['GEMINI_API_KEY']
)

# Chat model: Generates responses and evaluates answers
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,  # Low temperature for consistent, factual responses
    verbose=True
)

# ==========================================
# STEP 1: Build Q&A System
# ==========================================

# Load product catalog from CSV file
file = 'Products.csv'
loader = CSVLoader(file_path=file)

# Create searchable vector database from CSV
# Process: CSV rows → Text chunks → Vector embeddings → Searchable index
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,  # In-memory vector store (fast, temporary)
    embedding=embeddings  # Convert text to vectors for semantic search
).from_loaders([loader])

# Test the basic Q&A functionality
query = "Tell me which product I should get, I really enjoy listening to loud music"
response = index.query(query, llm=chat)
print("Basic Q&A test:", response[:100] + "...")

# Load documents for evaluation purposes
data = loader.load()  # Convert CSV to LangChain Document objects

# Create production-ready Q&A chain with better formatting
qa = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",  # "stuff" = combine all retrieved docs into prompt
    retriever=index.vectorstore.as_retriever(),  # Vector search retriever
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "\n--- PRODUCT ---\n"  # Clear product boundaries in context
    }
)

# ==========================================
# STEP 2: Question Generation (Two Approaches)
# ==========================================

# APPROACH A: LLM-Generated Questions (Automatic)
# This creates questions automatically from your documents
example_gen_chain = QAGenerateChain.from_llm(chat)

# Uncomment to use automatic question generation:
'''
print("=== AUTOMATIC QUESTION GENERATION ===")
auto_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:3]]  # Generate from first 3 products
)
print(f"Generated {len(auto_examples)} automatic Q&A examples")
'''

# APPROACH B: Manual Questions (Controlled)
# You create specific questions to test particular scenarios
print("=== MANUAL QUESTION CREATION ===")
new_examples = [
    {
        'query': 'What gaming console do you have?',
        'answer': 'Nintendo Switch OLED with a vibrant 7-inch OLED screen.'
        # ↑ Expected answer: What you think the correct answer should be
    },
    {
        'query': 'Which product is best for music?',
        'answer': 'The Bose QuietComfort 45 headphones are excellent for music listening.'
        # ↑ Testing music-related product recommendations
    },
    {
        'query': 'What laptops are available?',
        'answer': 'The MacBook Air M2 with longer battery life and thin design.'
        # ↑ Testing product listing functionality
    }
]

print(f"Created {len(new_examples)} manual Q&A examples for testing")

# ==========================================
# STEP 3: System Response Generation  
# ==========================================

# Get actual responses from your Q&A system
print("\n=== GENERATING SYSTEM RESPONSES ===")
predictions = qa.apply(new_examples)  # Run each question through your Q&A system

# What happens here:
# For each question → Vector search → Retrieve relevant products → LLM generates answer
print(f"Generated {len(predictions)} system responses")

# ==========================================
# STEP 4: LLM-as-Judge Evaluation
# ==========================================

# Create evaluation chain: This LLM will judge if answers are correct
print("\n=== LLM EVALUATION SETUP ===")
eval_chain = QAEvalChain.from_llm(chat)  # Same LLM acts as judge

# Evaluate system responses against expected answers
# The LLM compares: Expected answer vs System's actual answer
graded_outputs = eval_chain.evaluate(new_examples, predictions)

# What the evaluator LLM does:
# 1. Reads the question
# 2. Reads expected answer (your manual answer)
# 3. Reads system's actual answer  
# 4. Decides: CORRECT or INCORRECT

# ==========================================
# STEP 5: Results Analysis
# ==========================================

print("\n=== EVALUATION RESULTS ===")
correct_count = 0
total_count = len(new_examples)

for i, example in enumerate(new_examples):
    print(f"\nExample {i+1}:")
    print(f"Question: {predictions[i]['query']}")
    print(f"Expected Answer: {example['answer']}")  # Your manual expected answer
    print(f"System Answer: {predictions[i]['result']}")  # What your system actually said
    
    # Extract evaluation grade from LLM judge
    grade_text = graded_outputs[i]['results']  # Get judge's decision
    grade = grade_text.replace('GRADE: ', '')  # Clean up formatting
    print(f"LLM Judge Says: {grade}")
    
    # Count correct answers for accuracy calculation
    if "CORRECT" in grade:
        correct_count += 1
    
    print("-" * 50)

# ==========================================
# STEP 6: Performance Summary
# ==========================================

accuracy = (correct_count / total_count) * 100
print(f"\n=== PERFORMANCE SUMMARY ===")
print(f"Correct Answers: {correct_count}/{total_count}")
print(f"Accuracy: {accuracy:.1f}%")

# Interpretation guide:
print(f"\n=== EVALUATION INSIGHTS ===")
print("• High accuracy (>80%): Your Q&A system is working well")
print("• Medium accuracy (50-80%): Consider improving retrieval or prompts")  
print("• Low accuracy (<50%): Major issues with system or test questions")
print("\nNote: The LLM judge is evaluating based on semantic similarity,")
print("not exact string matching. It understands context and meaning.")
