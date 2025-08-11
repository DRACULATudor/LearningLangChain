"""
Simple Sequential Chain Tutorial
===============================
A Sequential Chain is combining multiple simple chains, where 
the output of a chain is the input of the next chain 

Simple Sequential Chain = single input/output
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
import pandas as pd
from dotenv import load_dotenv, find_dotenv

_=load_dotenv(find_dotenv())

lst_name = ['Tdor', 'Jake', 'Alexa', 'Maria']
lst_age = [21, 22, 30, 18]
lst_height = [1.93, 1.32, 1.90, 2.01]
data = [{'name': name, 'age': age, 'height': height} 
        for name, age, height in zip(lst_name, lst_age, lst_height)]

df = pd.DataFrame(data)
df.to_csv('Data.csv', index=False)


chat = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,
    verbose=True
)
prompt = ChatPromptTemplate.from_template('''Who is the tallest from these people in the data ?\n\n{people_data}\n\nProvide height.''')

prompt2 = ChatPromptTemplate.from_template('''From the previouse awnser:{text}\n\n, provide the person's name''')


# Note: LLMChain is deprecated but still works with SimpleSequentialChain
# Modern alternative would use RunnableSequence but is more complex
chain1 = LLMChain(llm=chat, prompt=prompt)
chain2 = LLMChain(llm=chat, prompt=prompt2)


#call the simpleseqchain on the chains, so basically we create
#the order of how the cains will be executed
people_data = "\n".join([f'{p["name"]}: {p["height"]}m tall' for p in data])


#here is where we actualy execute the two prompots in a chained order
#chain1 ouput is the input of chain2
overall_simple_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

#call the llm and pass the parsed data 
result = overall_simple_chain.run(people_data)
print(result)