"""
LangChain Conversation Buffer Memory Tutorial
=====================================
Demonstrates how to maintain conversation context across multiple LLM interactions
using ConversationBufferMemory to remember previous exchanges.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import warnings

warnings.filterwarnings('ignore')

_ = load_dotenv(find_dotenv()) # read local .env file

#preprare the llm
chat = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,
)

#prepare memory
memory = ConversationBufferMemory()

#configure conversation
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

#predict sends the input to the llm and automatically 
#stores both the input and the response
'''response = conversation.predict(input="Hi, my name is 2door")

print(response)

response = conversation.predict(input="Hi, what's my name ?")

print(response)'''


#IF YOU SET VERBOSE TO TRUE YOU CAN SEE THE PROMPT LANGCHAIN IS GENERATING
'''> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative 
and provides lots of specific details from its context. If the AI does not know the answer to a question, 
it truthfully says it does not know.

Current conversation:

Human: Hi, my name is 2door
AI:

> Finished chain.
Hi 2door! It's nice to meet you.  I'm an AI, and while I don't have a name in the same way humans do, 
you can think of me as a helpful and informative assistant.  I'm excited to chat with you. What's on your mind today? 
I have access to a vast amount of information, from historical facts and scientific data to current events and fictional narratives. 
I can even try to generate creative text formats like poems, code, scripts, musical pieces, email, letters, etc.  So, ask me anything!


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi, my name is 2door
AI: Hi 2door! It's nice to meet you.  I'm an AI, and while I don't have a name in the same way humans do, 
you can think of me as a helpful and informative assistant.  I'm excited to chat with you. What's on your mind today?  
I have access to a vast amount of information, from historical facts and scientific data to current events and fictional narratives.  
I can even try to generate creative text formats like poems, code, scripts, musical pieces, email, letters, etc.  So, ask me anything!
Human: Hi, what's my name ?
AI:

> Finished chain.
Hi 2door, you told me your name is 2door.  Is there anything else I can help you with?'''

#So as you can see we can see everything pormpted by langchain
#also see how the ai is able to remember my name

#if you want to check what s inisde the memoty buff
'print(memory.buffer)'

#and also to see what langchain remembered from the conversation
'print(memory.load_memory_variables({}))'

#this is how we can change and maiplaute the memory input/output 
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, {"output": "Wsup ?"})


print('adding to the fresh memory')
print(memory.buffer)
memory.load_memory_variables({})
print()

#so you can basically keep adding stuff to the memory
memory.save_context({"input": "Chillin and grillin"}, {"output": "What u grillin Boss ?"})

print('after adding to the memory')
print(memory.buffer)

print(memory.load_memory_variables({}))#this is how is actually stored

#WHY DO WE NEED MEMORY ?

#llm's don't actually have memory, so in order to provide an awnser
#a fresh api call needs to be made (it's going to be charged/token)
#so it can get quite pricy, even if it doesn't have to 

#That's why langachain offers different kind of tool for using the memory
#in such way so those prciy token api call dont have to be made with that much 
#When using the buffer the token usage grows with the convo length

print("\n=== SUMMARY ===")
print("ConversationBufferMemory stores the ENTIRE conversation history")
print("Pros: Perfect memory of all interactions")
print("Cons: Token usage grows linearly with conversation length")
print("Best for: Short conversations or when full context is needed")
