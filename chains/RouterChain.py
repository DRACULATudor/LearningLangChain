"""
Router Chain Tutorial
=====================
A Router Chain intelligently directs questions to specialized expert chains
based on the input content. It acts like a smart dispatcher that chooses
the best expert (Physics, Math, History, etc.) for each question.

Flow: Input → Router (decides) → Expert Chain → Response
"""

import os
from pprint import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains import LLMChain
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import warnings

warnings.filterwarnings('ignore')
_=load_dotenv(find_dotenv())

# Initialize the LLM that will power both routing decisions and expert responses
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ['GEMINI_API_KEY'],
    temperature=0.0,  # Low temperature for consistent routing decisions
    verbose=True
)

# ==========================================
# STEP 1: Define Expert Chain Templates
# ==========================================
# Each template creates a specialized "expert" with specific knowledge and personality

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

# ==========================================
# STEP 2: Configure Route Information
# ==========================================
# This defines which expert to use for what type of questions

prompt_infos = [
    {
        "name": "Physics",  # Must match destination name exactly
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "Math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "Computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    },
]

# ==========================================
# STEP 3: Build Expert Chains Dictionary
# ==========================================
# Convert each expert template into an actual LLM chain

destination_chain = {}  # Will store: {"Physics": physics_chain, "Math": math_chain, ...}

# Loop through each expert configuration and create its chain
for p_info in prompt_infos:
    name = p_info["name"]                    # Expert name (e.g., "Physics")
    prompt_template = p_info["prompt_template"]  # Expert's specialized prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=chat, prompt=prompt)
    destination_chain[name] = chain          # Store the expert chain

# ==========================================
# STEP 4: Create Router Instruction String
# ==========================================
# Build a human-readable list of available experts for the router

# Extract expert names and descriptions: ["Physics : Good for...", "Math : Good for..."]
destination = [f"{p['name']} : {p['description']}" for p in prompt_infos]

# Convert to single string for router template
destination_str = "\n".join(destination)

# ==========================================
# STEP 5: Setup Default Chain
# ==========================================
# Fallback chain for questions that don't fit any expert category

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=chat, prompt=default_prompt)

# ==========================================
# STEP 6: Create Router Decision Template
# ==========================================
# This template teaches the LLM how to choose the right expert

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: The value of "destination" MUST match one of \
the candidate prompts listed below.\
If "destination" does not fit any of the specified prompts, set it to "DEFAULT."
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

# ==========================================
# STEP 7: Build Router Chain
# ==========================================
# Inject available experts into router template

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destination_str  # Inserts the expert list
)

# Create router prompt with output parser to handle JSON response
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),  # Parses JSON to extract destination and input
)

# Create the router chain that makes routing decisions
router_chain = LLMRouterChain.from_llm(chat, router_prompt)

# ==========================================
# STEP 8: Assemble Complete Multi-Prompt Chain
# ==========================================
# Combine router + expert chains + default chain

chain = MultiPromptChain(
    router_chain=router_chain,           # Decides which expert to use
    destination_chains=destination_chain, # Dictionary of expert chains
    default_chain=default_chain,         # Fallback for unrecognized questions
    verbose=True                         # Show routing decisions
)

response = chain.run("what's the theory of relativity?")
print(response)
# Try other questions to test routing:
# print(chain.run("What is calculus?"))  # Should route to Math
# print(chain.run("Who was Napoleon?"))  # Should route to History
# print(chain.run("What is recursion?")) # Should route to Computer science