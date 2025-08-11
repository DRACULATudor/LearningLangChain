# 🦜⛓️ LearningLangChain

Follow me on my path to learning LangChain! Most of the resources are from DeepLearning.AI, this is my interpretation of it, hope it helps!

## 📚 Overview

This repository contains practical examples and tutorials for learning LangChain, focusing on conversation memory management, prompt templates, and output parsing. All examples use Google's Gemini AI model through LangChain's integration.

## 🗂️ Project Structure

```
LearningLangChain/
├── README.md
├── requirements.txt
├── .gitignore
├── agents/
│   └── agents.py                    # LangChain agents with tools
├── chains/
│   ├── llmChains.py                 # Basic LLM chains
│   ├── SImpleSequentialChain.py     # Simple sequential chain operations
│   ├── SequentialChain.py           # Advanced sequential chains
│   └── RouterChain.py               # Multi-prompt routing chains
├── evaluations/
│   └── evaluations.py               # Model evaluation and testing
├── memory/
│   ├── Conv_memorry_buffer.py       # Basic conversation buffer memory
│   ├── Conv_buff_window_mem.py      # Window-based conversation memory
│   ├── Conv_token_buff_mem.py       # Token-limited buffer memory
│   └── Conv_summ_memory.py          # Conversation summary memory
├── prompts_and_parsers/
│   ├── Prompt_templates.py          # Reusable prompt templates
│   └── Output_parse.py              # Structured output parsing
└── qnas/
    └── q&aOverDocs.py               # Question & Answer over documents
```

## 🚀 Features

### � Chain Operations
- **LLM Chains**: Basic language model chain operations
- **Sequential Chains**: Chain multiple LLM calls in sequence
- **Router Chains**: Intelligently route queries to appropriate specialized chains

### 🤖 Agents & Tools
- **ReAct Agents**: Reasoning and Acting agents with tool integration
- **Python REPL Tool**: Execute Python code dynamically
- **Tool Loading**: Integration with various external tools

### �💭 Memory Management
- **Buffer Memory**: Store complete conversation history
- **Window Memory**: Maintain only recent conversation turns
- **Token Buffer Memory**: Dynamically manage memory based on token usage
- **Summary Memory**: Compress old conversations into summaries

### 📝 Prompt Engineering
- **Template System**: Create reusable prompt templates with variables
- **Dynamic Formatting**: Inject different styles and content into templates
- **Consistent Interactions**: Maintain uniform LLM communication patterns

### 🔧 Output Processing
- **Structured Parsing**: Convert LLM responses into structured dictionaries
- **Schema Definition**: Define expected output formats
- **Data Validation**: Ensure outputs match predefined schemas

### 📚 Document Q&A
- **Document Loading**: Load and process CSV and other document formats
- **Vector Search**: Semantic search over document collections
- **Retrieval QA**: Question-answering over document collections

### 🧪 Evaluation & Testing
- **QA Generation**: Automatically generate test questions
- **Model Evaluation**: Assess model performance on Q&A tasks
- **Evaluation Chains**: Systematic evaluation workflows

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DRACULATudor/LearningLangChain.git
   cd LearningLangChain
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 🎯 Usage Examples

### Basic Conversation with Memory
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the model and memory
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=chat, memory=memory)

# Have a conversation
response = conversation.predict(input="Hi, my name is Andrew")
print(response)
```

### Using Prompt Templates
```python
from langchain.prompts import ChatPromptTemplate

template = """Translate the text into a style that is {style}.
text: ```{text}```"""

prompt_template = ChatPromptTemplate.from_template(template)
messages = prompt_template.format_messages(
    style="professional and polite",
    text="Your original text here"
)
```

### Structured Output Parsing
```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Define the output schema
schemas = [
    ResponseSchema(name="gift", description="Was this item purchased as a gift?"),
    ResponseSchema(name="delivery_days", description="How many days for delivery?"),
    ResponseSchema(name="price_value", description="Extract any sentences about price/value")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
```

## 📖 Learning Modules

### 1. Chains (`/chains/`)
- **`llmChains.py`**: Learn basic LLM chain operations and prompt handling
- **`SImpleSequentialChain.py`**: Understand simple sequential chain workflows
- **`SequentialChain.py`**: Master complex multi-step sequential chains
- **`RouterChain.py`**: Implement intelligent query routing systems

### 2. Agents (`/agents/`)
- **`agents.py`**: Explore ReAct agents with tool integration and Python execution

### 3. Memory Management (`/memory/`)
- **`Conv_memorry_buffer.py`**: Learn basic conversation memory
- **`Conv_buff_window_mem.py`**: Understand window-based memory limitations
- **`Conv_token_buff_mem.py`**: Explore token-efficient memory management
- **`Conv_summ_memory.py`**: Implement conversation summarization

### 4. Prompts & Parsing (`/prompts_and_parsers/`)
- **`Prompt_templates.py`**: Master reusable prompt creation
- **`Output_parse.py`**: Transform unstructured responses into structured data

### 5. Document Q&A (`/qnas/`)
- **`q&aOverDocs.py`**: Build question-answering systems over document collections

### 6. Evaluation (`/evaluations/`)
- **`evaluations.py`**: Learn to evaluate and test LLM applications systematically

## 🔐 Security Best Practices

✅ **This project follows secure API key management:**
- Environment variables for API keys
- No hardcoded secrets in source code
- `.env` file usage (remember to add it to `.gitignore`)

## 🛠️ Technologies Used

- **LangChain**: Framework for developing applications with LLMs
- **Google Gemini**: Advanced AI model for text generation
- **Python-dotenv**: Environment variable management
- **Python 3.8+**: Programming language

## 📋 Requirements

```
langchain>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.0.20
langchain-experimental>=0.0.50
python-dotenv>=1.0.0
pandas>=1.3.0
ipython>=7.0.0
langchain-hub>=0.1.0
google-generativeai>=0.3.0
```

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements or additional examples!

## 📄 License

This project is open source

## 🙏 Acknowledgments

- [DeepLearning.AI](https://www.deeplearning.ai/) for the excellent LangChain course content
- [LangChain](https://python.langchain.com/) for the amazing framework
- [Google](https://ai.google.dev/) for the Gemini API

---

**Happy Learning! 🎓**
