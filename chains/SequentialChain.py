"""
Sequential Chain Tutorial
===============================
A Sequential Chain is combining multiple simple chains, where 
the output of a chain is the input of the next chain 

Sequential Chain = multiple inputs/outputs
"""


import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import warnings

warnings.filterwarnings('ignore')
_=load_dotenv(find_dotenv())

#here well use data1.csv since we want to make the chain response more obvious

df = pd.read_csv('Data1.csv')
print(df.head())


chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,
    verbose=True
)

first_prompt = ChatPromptTemplate.from_template(
    "Translate the following description to english:"
    "\n\n{Description}"
)

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following description in 1 sentence:"
    "\n\n{English_Description}"
)
#X

trd_prompt = ChatPromptTemplate.from_template(
    "Can you tell me in what languages the initial description was made ?"
    "\n\n{Description}"
)

fth_prompt = ChatPromptTemplate.from_template(
    "Can you write a better, shorter, description in the original language, using the summary?"
    "\n\nSummary: {summary}\n\nLanguage:{language}\n\n"
)


#output key will the the llmchain what variable to us when toring ai s response
chain_one = LLMChain(llm=chat, prompt=first_prompt, output_key="English_Description")
                                                                #X
                                                                #the variable names MUST MATCH ouput keys
chain_two = LLMChain(llm=chat, prompt=second_prompt, output_key="summary")

chain_three = LLMChain(llm=chat, prompt=trd_prompt, output_key="language")

chain_four = LLMChain(llm=chat, prompt=fth_prompt, output_key="improved_description")


#call the sequential chin and set it up
#shows how all the chain will be run, and what ouput keys to look after
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Description"],
    output_variables=["English_Description", "summary", "language", "improved_description"],
    verbose=True
)

#extract first row from csv into a string
data = df.iloc[0]['description'] 

#create the input key for the llm
response = overall_chain.run({"Description" : data})

print(response)


