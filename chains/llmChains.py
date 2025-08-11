"""
LangChain Simple Chain Tutorial
===============================
This how to create basic LLM chains using prompt templates
and modern pipe operator syntax for data analysis tasks.

Data → Prompt → LLM → Answer
"""


import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

# Configure Gemini 
# API


#run this first to generate some data, modify it if you want, or add more elems
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

prompt = ChatPromptTemplate.from_template('''Who is the tallest from these people in the data ?\n\n{people_data}\n\n
                                          Provide name, age, and height.''')

# Use modern syntax instead of deprecated LLMChain
'''chain = LLMChain(llm=chat, prompt=prompt) <-this wo'''

#LLMchain has been removed for the pipe operator

#this is the new way of creating a chain
chain = prompt | chat

# Important the llm expects the data as a human readable string

#here are some ways

people_data = " | ".join([f'{p["name"]} ({p["height"]}m)' for p in data])
# Result: Tdor (1.93m) | Jake (1.32m) | Alexa (1.90m) | Maria (2.01m)

# Option 2: Comma separated
people_data = ", ".join([f'{p["name"]}: {p["height"]}m' for p in data])
# Result: Tdor: 1.93m, Jake: 1.32m, Alexa: 1.90m, Maria: 2.01m

# Option 3: Bullet points with newlines
people_data = "\n".join([f'• {p["name"]}: {p["height"]}m tall' for p in data])
# Result:
# • Tdor: 1.93m tall
# • Jake: 1.32m tall
# • Alexa: 1.90m tall
# • Maria: 2.01m tall

# Option 4: Table format
people_data = f"Name | Age | Height\n" + "\n".join([f'{p["name"]} | {p["age"]} | {p["height"]}m' for p in data])


#we pass as dict since this is what the template is expecting
response = chain.invoke({'people_data' : people_data})

print(response.content)