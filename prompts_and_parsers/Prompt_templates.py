"""
LangChain Prompt Templates Tutorial
==================================
Demonstrates how to create reusable prompt templates with variables
for consistent LLM interactions.
"""


import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

# Configure Gemini API

chat = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,
)
if __name__ == "__main__":

    #defining the template
    template_string = """Translate the text \
    that is delimited by triple backticks \
    into a style that is {style}. \
    text: ```{text}```
    """

    #style
    customer_style = """American English \
    in a calm and respectful tone
    """

    #text
    customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse, \
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
    """

    #getting a prompt var as the tempplate
    prompt_template = ChatPromptTemplate.from_template(template_string)

    #basically this is where it all gets combined
    customer_messages = prompt_template.format_messages(
        style=customer_style,
        text=customer_email
    )

    #now call the actual llm on the prompt with the desired text and styling
    customer_response = chat(customer_messages)
    '''print(customer_response.content) <--uncoment if you wanna test'''

    #OUTPUT
    service_reply = """Hey there customer, \ <--- THIS IS WHAT WILL BE RETURNED
    the warranty does not cover \
    cleaning expenses for your kitchen \
    because it's your fault that \
    you misused your blender \
    by forgetting to put the lid on before \
    starting the blender. \
    Tough luck! See ya!
    """
    