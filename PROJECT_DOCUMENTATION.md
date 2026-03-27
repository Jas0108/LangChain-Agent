# LangChain ReAct Agent - Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [What is ReAct?](#what-is-react)
3. [Architecture & Flow](#architecture--flow)
4. [Project Structure](#project-structure)
5. [File-by-File Explanation](#file-by-file-explanation)
6. [The ReAct Prompt — Deep Dive](#the-react-prompt--deep-dive)
7. [How the Tools Work](#how-the-tools-work)
8. [Design Decisions & Justifications](#design-decisions--justifications)
9. [Sample Test Results](#sample-test-results)
10. [Tech Stack](#tech-stack)

---

## Project Overview

This project is a **LangChain-based ReAct AI Agent** that can intelligently decide whether to use an external tool or answer a question directly using its own knowledge. It uses **Ollama's llama3.2:3b** model running locally and has access to 3 specialized tools:

- **Currency Conversion** — Converts money between currencies using the ExchangeRate API
- **Language Translation** — Translates English text to other languages using the free MyMemory API
- **Calculator** — Evaluates math expressions using Python's `math` library

The agent uses the **ReAct (Reasoning + Acting)** framework — it thinks step-by-step before deciding whether to use a tool or answer directly.

---

## What is ReAct?

**ReAct** stands for **Reasoning + Acting**. It is a prompting strategy where the LLM is forced to follow a structured loop:

```
Thought:  → The model reasons about what to do
Action:   → The model picks a tool to use
Action Input: → The model provides the input for that tool
Observation:  → The tool returns a result
Thought:  → The model reflects on the result
Final Answer: → The model gives the final response
```

If the model decides **no tool is needed**, it skips the Action/Observation loop and goes straight to:

```
Thought: This is a general knowledge question. No tool needed.
Final Answer: [direct answer]
```

**Why ReAct?** It forces the model to **think before acting**. The model must explicitly write a "Thought" explaining its reasoning before it can call any tool. This makes tool routing more deliberate and transparent.

---

## Architecture & Flow

```
User Question
     |
     v
+----------------------------+
|  ReAct Agent (LLM + Tools) |
|                            |
|  Thought: [reasoning]      |
|     |                      |
|     |-- Tool needed?       |
|     |   YES → Action       |
|     |         Action Input |
|     |         Observation  |
|     |         Final Answer |
|     |                      |
|     |   NO → Final Answer  |
+----------------------------+
     |
     v
  AI Response
```

### The ReAct Loop

1. The LLM receives the user's question along with tool descriptions
2. It writes a **Thought** — reasoning about what the question is asking
3. If a tool is needed, it writes **Action** (tool name) and **Action Input** (tool parameters)
4. The AgentExecutor runs the tool and feeds the **Observation** (result) back to the LLM
5. The LLM writes another **Thought** reflecting on the observation, then gives a **Final Answer**
6. If no tool is needed, the LLM skips Action/Observation and goes straight to **Final Answer**

---

## Project Structure

```
LangChain Agent/
├── main.py           # LLM setup, ReAct prompt, agent executor, chat loop
├── tools.py          # 3 tool definitions (currency, translation, calculator)
├── config.py         # API keys
├── requirements.txt  # Python dependencies
└── database.py       # Legacy file (not used in current version)
```

---

## File-by-File Explanation

### 1. config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")
```

**Purpose:** Loads API keys from a `.env` file using `python-dotenv`, keeping secrets out of the codebase.

- `load_dotenv()` — Reads key-value pairs from the `.env` file and sets them as environment variables
- `os.getenv("CURRENCY_API_KEY")` — Retrieves the key from the environment
- The `.env` file is listed in `.gitignore` so it never gets pushed to GitHub
- The translation tool (MyMemory API) does not need an API key — it's free
- The calculator tool does not need an API key — it uses Python's built-in `math` library

---

### 2. tools.py

```python
import json
import math
import requests
from langchain_core.tools import tool
import config
```

**Purpose:** Defines the 3 tools the agent can use. Each tool is a Python function decorated with `@tool` from LangChain, which makes it callable by the ReAct agent.

#### Tool 1: `convert_currency(query)`

```python
@tool
def convert_currency(query: str) -> str:
    """Use this tool ONLY when the user wants to convert or exchange money from one
    currency to another. Supports all major currencies like USD, INR, EUR, GBP, JPY, etc.
    Input must be a JSON string with three keys: amount (number), from_currency (currency code),
    to_currency (currency code).
    Example input: {"amount": 100, "from_currency": "USD", "to_currency": "INR"}"""
```

**How it works:**
1. Receives a single JSON string as input (e.g. `{"amount": 100, "from_currency": "USD", "to_currency": "INR"}`)
2. Parses the JSON to extract `amount`, `from_currency`, and `to_currency`
3. Calls the ExchangeRate-API to get the latest conversion rate
4. Multiplies `amount × rate` and returns a formatted result like: `"100.0 USD = 8403.00 INR"`

**Why single string input?** The ReAct agent passes `Action Input` as a single string. Multi-parameter tools break because the ReAct parser can't split the string into separate arguments. Accepting one JSON string and parsing internally solves this.

**API Used:** ExchangeRate-API (https://www.exchangerate-api.com/) — requires an API key stored in `config.py`

#### Tool 2: `translate_text(query)`

```python
@tool
def translate_text(query: str) -> str:
    """Use this tool ONLY when the user wants to translate English text into another language.
    Supports many languages. Input must be a JSON string with two keys: text (the English text
    to translate), to_language (language code). Common language codes: hi (Hindi), es (Spanish),
    fr (French), de (German), ja (Japanese), zh (Chinese), ar (Arabic), ko (Korean).
    Example input: {"text": "hello", "to_language": "hi"}"""
```

**How it works:**
1. Receives a single JSON string as input (e.g. `{"text": "hello", "to_language": "hi"}`)
2. Parses JSON to extract `text` and `to_language`
3. Sends a GET request to MyMemory API with the text and language pair (e.g. `en|hi`)
4. Extracts the translated text from the JSON response
5. Returns a formatted string like: `"Translation: नमस्ते"`

**API Used:** MyMemory Translation API (https://mymemory.translated.net/) — completely free, no API key needed, supports ~5000 words/day

**Supported language codes:** hi (Hindi), es (Spanish), fr (French), de (German), ja (Japanese), zh (Chinese), ar (Arabic), ko (Korean), and many more

#### Tool 3: `calculator(expression)`

```python
@tool
def calculator(expression: str) -> str:
    """Use this tool ONLY when the user asks to calculate, compute, or solve a math expression.
    Input must be a plain math expression as a string. Supports: +, -, *, /, ** (power),
    sqrt(), abs(), round(), log(), sin(), cos(), tan(), pi, e.
    Example inputs: 2+2, 100/4, sqrt(16), 2**10, 3.14*5"""
```

**How it works:**
1. Receives a plain math expression as a string (e.g. `"2+2"`, `"sqrt(16)"`, `"100/4"`)
2. Evaluates it using Python's `eval()` with a restricted environment for safety
3. Only math functions are allowed (`sqrt`, `pow`, `abs`, `round`, `log`, `sin`, `cos`, `tan`, `pi`, `e`)
4. Built-in Python functions are blocked (`__builtins__` is set to empty) to prevent code injection
5. Returns a formatted result like: `"2+2 = 4"` or `"sqrt(16) = 4.0"`

**No API needed** — this tool runs entirely locally using Python's `math` library.

#### The `@tool` Decorator

The `@tool` decorator from LangChain does the following:
- Converts the function into a LangChain Tool object
- Uses the function's **docstring** as the tool description (the LLM reads this to decide when and how to use it)
- Uses the function's **type hints** (e.g. `query: str`) to define the expected input schema
- Makes the function callable by the AgentExecutor

**Tool descriptions are critical in ReAct** — the LLM reads them inside the `{tools}` placeholder in the prompt to understand what each tool does and what input format it expects.

---

### 3. main.py

This is the main file that ties everything together. It has 5 logical steps:

#### Step 1: Create the LLM

```python
llm = OllamaLLM(model="llama3.2:3b", temperature=0)
```

- `OllamaLLM` connects to a locally running Ollama instance (text completion mode)
- `model="llama3.2:3b"` — uses the Llama 3.2 model with 3 billion parameters
- `temperature=0` — makes the model deterministic (same input = same output, no randomness)

**Why `OllamaLLM` instead of `ChatOllama`?** ReAct agents work with **text completion** models, not chat models. `OllamaLLM` produces raw text output that follows the Thought/Action/Final Answer format. `ChatOllama` is designed for chat-based tool calling (a different approach).

#### Step 2: The ReAct Prompt

```python
react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can.
You have access to the following tools:

{tools}

TOOL ROUTING RULES:
- USE convert_currency ONLY for converting/exchanging money between currencies
- USE translate_text ONLY for translating text to another language
- USE calculator ONLY for math calculations or arithmetic
- DO NOT use any tool for general knowledge, greetings, opinions, or definitions.
  Give Final Answer directly.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

EXAMPLES:
[few-shot examples showing both tool usage and direct answers]

Begin!

Question: {input}
Thought:{agent_scratchpad}""")
```

**This prompt has 5 key sections:**

1. **Tool descriptions** (`{tools}`) — LangChain auto-fills this with each tool's name and docstring
2. **Routing rules** — Explicit instructions on WHEN to use each tool and when NOT to
3. **ReAct format** — The Thought → Action → Observation → Final Answer template
4. **Few-shot examples** — Concrete examples showing the model exactly how to behave
5. **Placeholders** — `{input}` for the user's question, `{agent_scratchpad}` for the agent's ongoing work

**Key variables in the prompt:**
- `{tools}` — Replaced by LangChain with tool names + docstrings
- `{tool_names}` — Replaced with comma-separated tool names (e.g. `convert_currency, translate_text, calculator`)
- `{input}` — Replaced with the actual user question
- `{agent_scratchpad}` — Where LangChain appends previous Thought/Action/Observation steps during multi-step reasoning

#### Step 3: Create the Agent and Executor

```python
tools = [convert_currency, translate_text, calculator]
agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    return_intermediate_steps=True
)
```

- `create_react_agent(llm, tools, react_prompt)` — Creates a ReAct agent that parses the LLM's text output for "Action:" and "Action Input:" lines, then executes the matching tool
- `AgentExecutor` — Runs the ReAct loop: LLM thinks → calls tool → gets observation → LLM thinks again → final answer
  - `verbose=True` — Shows the full Thought/Action/Observation chain (useful for debugging)
  - `handle_parsing_errors=True` — Gracefully handles cases where the LLM outputs malformed text
  - `max_iterations=5` — Prevents infinite loops (stops after 5 tool calls max)
  - `return_intermediate_steps=True` — Returns the raw tool outputs so we can use them directly

#### Step 4: Handle the Response

```python
def ask_agent(user_input):
    response = executor.invoke({"input": user_input})

    for step in response.get("intermediate_steps", []):
        tool_output = step[1]
        if tool_output and "error" not in tool_output.lower():
            return tool_output

    return response["output"]
```

**Why extract tool output directly?** The small LLM sometimes ignores, rephrases, or contradicts the tool's result in its Final Answer. By grabbing the raw tool output from `intermediate_steps`, we guarantee the user sees the exact, correct answer from the tool. If no tool was called (direct answer), we fall back to `response["output"]` which contains the Final Answer.

#### Step 5: Chat Loop

```python
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    if not user_input:
        continue

    answer = ask_agent(user_input)
    print(f"\nAI: {answer}\n")
```

- Simple loop that reads user input, sends it to the agent, and prints the answer
- Typing "quit", "exit", or "q" exits the program
- Empty input is skipped

---

### 4. requirements.txt

```
langchain
langchain-community
langchain-core
langchain-ollama
langgraph
requests
```

- `langchain` — Core LangChain framework
- `langchain-community` — Community integrations (includes langchain_classic for agents)
- `langchain-core` — Core components (prompts, tools)
- `langchain-ollama` — Ollama integration for running local LLMs
- `langgraph` — Graph-based agent framework (dependency)
- `requests` — HTTP library for making API calls in tools

---

## The ReAct Prompt — Deep Dive

### Why Prompt Engineering Matters

The entire intelligence of this agent lives in the prompt. The LLM itself is a general-purpose text generator — the prompt is what turns it into a tool-routing agent. Every section of the prompt serves a specific purpose:

### Section 1: Tool Descriptions

```
{tools}
```

LangChain automatically replaces `{tools}` with each tool's name and docstring. This is why the docstrings in `tools.py` are so detailed — the LLM reads them to understand:
- **When** to use each tool (e.g. "ONLY when the user wants to convert money")
- **What input format** is expected (e.g. "JSON string with three keys")
- **Example input** (e.g. `{"amount": 100, "from_currency": "USD", "to_currency": "INR"}`)

### Section 2: Routing Rules

```
TOOL ROUTING RULES:
- USE convert_currency ONLY for converting/exchanging money
- USE translate_text ONLY for translating text
- USE calculator ONLY for math calculations
- DO NOT use any tool for general knowledge, greetings, opinions, or definitions.
```

These rules act as **guardrails**. They tell the model explicitly:
- Which tool maps to which type of question
- What categories of questions should **not** use any tool

### Section 3: ReAct Format Template

```
Question: ...
Thought: ...
Action: ...
Action Input: ...
Observation: ...
Thought: I now know the final answer
Final Answer: ...
```

This is the standard ReAct format. It defines a **strict output structure** that LangChain's parser can extract tool calls from. The `create_react_agent` parser looks for:
- `Action:` followed by a tool name
- `Action Input:` followed by the tool input
- `Final Answer:` followed by the response to return to the user

### Section 4: Few-Shot Examples

The prompt includes concrete examples showing:

1. **General knowledge question → No tool (direct answer):**
   ```
   Question: What is glucose?
   Thought: This is a general knowledge question. I can answer directly without any tool.
   Final Answer: Glucose is a simple sugar...
   ```

2. **Math question → Calculator tool:**
   ```
   Question: What is 5 + 3?
   Thought: The user wants to do math. I should use calculator.
   Action: calculator
   Action Input: 5+3
   Observation: 5+3 = 8
   Thought: I now know the final answer
   Final Answer: 5 + 3 = 8
   ```

3. **Currency question → Currency tool:**
   ```
   Action: convert_currency
   Action Input: {"amount": 50, "from_currency": "USD", "to_currency": "EUR"}
   ```

4. **Translation question → Translation tool:**
   ```
   Action: translate_text
   Action Input: {"text": "goodbye", "to_language": "es"}
   ```

**Why few-shot examples?** Small LLMs learn best by example. The few-shot examples show the model the exact format to follow, including when to skip tools entirely. The calculator example is especially important because it shows the full Thought → Action → Observation → Thought → Final Answer cycle.

---

## How the Tools Work

### Tool Execution Flow (ReAct)

```
User: "Convert 100 USD to INR"
  |
  v
LLM writes: Thought: The user wants to convert currency. I should use convert_currency.
LLM writes: Action: convert_currency
LLM writes: Action Input: {"amount": 100, "from_currency": "USD", "to_currency": "INR"}
  |
  v
AgentExecutor parses "Action: convert_currency" and "Action Input: {...}"
  |
  v
Calls convert_currency() function in tools.py
  |
  v
Function calls ExchangeRate-API, returns: "100.0 USD = 8403.00 INR"
  |
  v
AgentExecutor feeds back: Observation: 100.0 USD = 8403.00 INR
  |
  v
LLM writes: Thought: I now know the final answer
LLM writes: Final Answer: 100.0 USD = 8403.00 INR
  |
  v
We extract the tool output from intermediate_steps and return it
```

### Direct Answer Flow (No Tool)

```
User: "What is glucose?"
  |
  v
LLM writes: Thought: This is a general knowledge question. No tool needed.
LLM writes: Final Answer: Glucose is a simple sugar and the primary source of energy...
  |
  v
AgentExecutor sees "Final Answer:" and returns the text after it
```

---

## Design Decisions & Justifications

| Decision | Why |
|----------|-----|
| **ReAct (create_react_agent)** | Forces the model to think (Thought) before acting (Action). The text-based format is more transparent and debuggable than native tool calling. |
| **OllamaLLM (text completion)** | ReAct agents need text completion to output the Thought/Action/Final Answer format. Chat models (ChatOllama) are designed for structured tool calling, which is a different approach. |
| **Ollama + llama3.2:3b** | Runs locally (no cloud API costs). Supports both ReAct text generation and tool calling. |
| **Single-string tool inputs** | ReAct's parser passes Action Input as one string. Multi-parameter tools break because the parser can't split into separate arguments. Accepting JSON strings and parsing internally is the standard fix. |
| **Extracting tool output from intermediate_steps** | The LLM sometimes rephrases or contradicts the tool's result. Grabbing the raw tool output ensures the user sees the exact correct answer. |
| **Few-shot examples in prompt** | Small LLMs learn best from examples. Showing both "use tool" and "don't use tool" patterns teaches correct routing behavior. |
| **Routing rules in prompt** | Explicit "USE X for Y" and "DO NOT use for Z" rules act as guardrails to prevent the model from calling tools unnecessarily. |
| **Calculator via safe eval()** | Uses Python's `eval()` with `__builtins__` disabled and only `math` functions allowed. Prevents code injection while supporting complex expressions. |
| **AgentExecutor with max_iterations=5** | Prevents infinite loops where the agent keeps calling tools without reaching an answer. |
| **MyMemory Translation API** | Free, no API key needed, reliable. Supports many languages out of the box. |
| **ExchangeRate-API** | Free tier available, provides real-time exchange rates for 160+ currencies. |
| **temperature=0** | Makes the LLM deterministic — same question always gives same response. Critical for reliable tool routing. |

---

## Sample Test Results

```
You: what is 2 + 2 + 4?
Thought: The user wants to do math. I should use calculator.
Action: calculator
Action Input: 2+2+4
Observation: 2+2+4 = 8
AI: 2+2+4 = 8

You: what is glucose
Thought: This is a general knowledge question. No tool needed.
Final Answer: Glucose is a simple sugar and the primary source of energy for the body's cells.
AI: Glucose is a simple sugar and the primary source of energy for the body's cells.

You: convert 100 USD to INR
Thought: The user wants to convert currency. I should use convert_currency.
Action: convert_currency
Action Input: {"amount": 100, "from_currency": "USD", "to_currency": "INR"}
Observation: 100.0 USD = 8403.00 INR
AI: 100.0 USD = 8403.00 INR

You: translate hello to Hindi
Thought: The user wants to translate text. I should use translate_text.
Action: translate_text
Action Input: {"text": "hello", "to_language": "hi"}
Observation: Translation: नमस्ते
AI: Translation: नमस्ते
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama + llama3.2:3b (local) |
| Agent Framework | LangChain ReAct (create_react_agent + AgentExecutor) |
| Prompt Strategy | ReAct (Reasoning + Acting) with few-shot examples |
| Currency API | ExchangeRate-API |
| Translation API | MyMemory Translation API |
| Calculator | Python math library (safe eval) |
| Language | Python 3 |
| HTTP Client | requests |

---

## How to Run

1. Install Ollama and pull the model: `ollama pull llama3.2:3b`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python main.py`
