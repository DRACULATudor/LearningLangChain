"""
LangChain Conversation Summary Buffer Memory Tutorial
=====================================
Demonstrates how to maintain conversation context using AI summarization
when conversations exceed token limits.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
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

#ConversationSummaryBufferMemory is smart memory, that uses ai for summarizing
#so it respects the token limit 
memory = ConversationSummaryBufferMemory(llm=chat, max_token_limit=50)
convo=ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

print(memory.load_memory_variables({}))