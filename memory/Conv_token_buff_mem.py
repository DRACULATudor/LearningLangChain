"""
LangChain Conversation Token Buffer Memory Tutorial
=====================================
Demonstrates how to maintain conversation context across multiple LLM interactions
using ConversationTokenBufferMemory to dinamically remember on tokens usage.
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
import warnings

warnings.filterwarnings('ignore')

_ = load_dotenv(find_dotenv()) # read local .env file

'''Temperature Scale (0.0 - 1.0):
0.0 = Deterministic, consistent, factual responses
0.1 = Very focused, minimal creativity
0.5 = Balanced creativity and consistency
1.0 = Maximum creativity, unpredictable responses'''

chat = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,
)


#basically token memory is definig in the memory how many tokens can be held in the history
#play with the token size if you want and check the history being turncated as the tokens decrease
#or add more and increase the token size for more memory
memory=ConversationTokenBufferMemory(llm=chat, max_token_limit=200)


conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

memory.save_context({"input": "Hi"}, {"output": "Wsup?"})
memory.save_context({"input": "Good, hbu ?"}, {"output": "u know aiin' and vibin'"})
memory.save_context({"input": "Lool, u've learnt some slang"}, {"output": "We will take over.... oops I mean,  u right boi :)"})
print(memory.load_memory_variables({}))