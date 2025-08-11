"""
LangChain Agents Tutorial
========================
Demonstrates how to create AI agents that can use tools to solve problems.
"""

# Fixed imports for newer LangChain versions
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools  # ✅ Updated import
from langchain_experimental.tools import PythonREPLTool  # ✅ Fixed import
from langchain import hub
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

# ==========================================
# SETUP: Initialize AI Models
# ==========================================

# Chat model: Generates responses and evaluates answers
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,  # Low temperature for consistent, factual responses
    verbose=True
)

# ==========================================
# AGENT 1: Math Calculator Agent
# ==========================================

print("=== MATH CALCULATOR AGENT ===")

# Load math calculation tools
tools = load_tools(["llm-math"], llm=chat)

# Get a prompt template for ReAct agents
try:
    prompt = hub.pull("hwchase17/react")
    print("✅ Successfully loaded ReAct prompt from hub")
except Exception as e:
    print(f"⚠️ Hub not accessible ({e}), using fallback prompt")
    # Fallback prompt if hub is not accessible
    from langchain.prompts import PromptTemplate
    
    prompt = PromptTemplate.from_template("""
    You are an assistant that can solve problems step by step.
    You have access to tools to help you.
    
    TOOLS:
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Question: {input}
    {agent_scratchpad}
    """)

# Create the ReAct agent
agent = create_react_agent(chat, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)

# Test the math agent
print("\n🧮 Testing Math Agent...")
result = agent_executor.invoke({"input": "What is 25% of 300?"})
print("Math Result:", result['output'])

# ==========================================
# AGENT 2: Python Code Execution Agent  
# ==========================================
'''
print("\n=== PYTHON EXECUTION AGENT ===")

try:
    # Create Python REPL tool for code execution
    #repl tool is baisaclly the way you tell ythe llm to compile the code generated
    python_tool = PythonREPLTool()
    python_tools = [python_tool]

    # Create Python agent executor
    python_agent_executor = AgentExecutor(
        agent=create_react_agent(chat, python_tools, prompt),
        tools=python_tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # Test Python code execution
    print("\n🐍 Testing Python Agent...")
    code_result = python_agent_executor.invoke({
        "input": "Create a list of squares from 1 to 10 and calculate their sum"
    })
    print("Python Code Result:", code_result['output'])

except ImportError as e:
    print(f"⚠️ Python REPL tool not available: {e}")
    print("Skipping Python agent demo...")
'''