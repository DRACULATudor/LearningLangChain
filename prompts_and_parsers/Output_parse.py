"""
LangChain Output Parsing Tutorial
=================================
Demonstrates how to convert LLM string outputs into structured dictionaries
using ResponseSchema and StructuredOutputParser.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv, find_dotenv

    #HOW TO PARSE THE OUPUT HOWEVER YOU WANT
    #we can also decide how we want the ouput given back
    #{
    #  "gift": False,
    #  "delivery_days": 5,
    #  "price_value": "pretty affordable!"
    #}

_ = load_dotenv(find_dotenv()) # read local .env file

# Configure Gemini API

chat = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,
)


customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

#an example of how we would like the awnser back
review_template = """\
For the following text, extract the following information:
gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.
delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.
price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.
Format the output as JSON with the following keys:
gift
delivery_days
price_value
text: {text}
"""


'''prompt_template = ChatPromptTemplate.from_template(review_template)'''
'''customer_review = prompt_template.format_messages(text=customer_review)'''

'''reply = chat(customer_review) <--unccoment for testing
print(reply.content)
'''
#THIS IS WHAT COMES AS REPLY
'''json
{
  "gift": false,
  "delivery_days": -1,
  "price_value": []
}
'''

#HOWEVER IF WE PRINT THE TYPE WE NOTICE IT'S A STRING NOT A DICT 
'''print(type(reply.content))'''

#IF WE WOULD TRY TO GET A VAL FROM THE JSON
'''print(reply.content.get('gift'))''' #<- this would cause an error since it's a string

#HOW TO CONVERT THE STRING INTO A DICT

#First create a Response schema

#define what to be expected as gift, delivery, price (as key, value)
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased as a gift for someone else? " \
                             "Answer True if yes, False if not or unknown.")


delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days did it take for the product to arrive? " \
                                      "If this information is not found, output -1.")

price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any sentences about the value or price,"
                                    " and output them as a comma separated Python list.")

#add them all to a list
response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]

#check what is now expected

#firt give the desired structure to the parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
'''print(output_parser) check the list'''
#than let's check to ensure that everything worked by checkign the parser_instructions

parser_instructions = output_parser.get_format_instructions()
print(parser_instructions)
#will output 
'''json
{
        "gift": string  // Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.
        "delivery_days": string  // How many days did it take for the product to arrive? If this information is not found, output -1.
        "price_value": string  // Extract any sentences about the value or price, and output them as a comma separated Python list.
}
'''
#so as you see it worked, now we have it all strcutred as we wanted

#now we also add the formatting instucrions 
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

#define prompting template
prompt_template = ChatPromptTemplate.from_template(template=review_template_2)

#pass the formating instructinos into the mssg
mssg =  prompt_template.format_messages(text=customer_review, format_instructions=parser_instructions)

#pass the mssg into the llm
respone = chat(mssg)

#now our output will be a dicionary
output_dict = output_parser.parse(respone.content)

#Now we succesefuly converted the output into a dict and can safely extract the gift val
output_dict.get('gift')

#BY Doing this we can now convert the ouput of the llm into any data, for diff usses