"""
LangChain Conversation Buffer Window Memory Tutorial
=====================================
Demonstrates how to maintain conversation context across multiple LLM interactions
using ConversationBufferWindowMemory to remember a fixed number of exchnges(ex. k=4).
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import warnings

warnings.filterwarnings('ignore')

_ = load_dotenv(find_dotenv()) # read local .env file

#preprare the llm

#Temperature controls the randomness/creativity of the AI's responses:

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

#prepare memory

#notice what happens when we set k to 1
memory = ConversationBufferWindowMemory(k=1)

#configure conversation
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

response = conversation.predict(input="My name is Tdor")
print(response)
response = conversation.predict(input="what's 2 + 2 ?")
print(response)
response = conversation.predict(input="What's my name")
print(response)

#SO AS YOU CAN SEE THE AI IS NO LONGER ABLE TO REMEMBER THE NAME
#THAT'S BEACUSE OF THE MEMORY BUFFER WINDOW, WHICH BY SETTING K TO A DESIRED VALUE
#IS THE NUMBER OF EXCHANGES THE AI WILL REMEMBER


#BTW THIS IS WHAT VERBOSE WILL SHOW
'''> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: My name is Tdor
AI:

> Finished chain.
It's nice to meet you, Tdor


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: My name is Tdor
AI: It's nice to meet you, Tdor
Human: what's 2 + 2 ?
AI:

> Finished chain.
2 + 2 = 4.  That


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: what's 2 + 2 ?
AI: 2 + 2 = 4.  That
Human: What's my name
AI:

> Finished chain.
I do not know your name.  I have'''