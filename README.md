# LangChain ReAct Agent

A Python-based AI agent built with **LangChain** and **Ollama** that uses the **ReAct (Reasoning + Acting)** framework to intelligently route user queries to the right tool or answer directly when no tool is needed.

## Tools

| Tool | Description | API |
|------|-------------|-----|
| **Currency Converter** | Converts money between 160+ currencies | [ExchangeRate-API](https://www.exchangerate-api.com/) |
| **Language Translator** | Translates English text to 20+ languages | [MyMemory API](https://mymemory.translated.net/) (free, no key) |
| **Calculator** | Evaluates math expressions (supports sqrt, log, trig, etc.) | Python `math` library (no API) |

## How It Works

The agent uses the **ReAct** prompting strategy — it thinks before it acts:

```
You: Convert 100 USD to INR

Thought: The user wants to convert currency. I should use convert_currency.
Action: convert_currency
Action Input: {"amount": 100, "from_currency": "USD", "to_currency": "INR"}
Observation: 100.0 USD = 8403.00 INR

AI: 100.0 USD = 8403.00 INR
```

For general questions, it skips tools entirely:

```
You: What is glucose?

Thought: This is a general knowledge question. No tool needed.
Final Answer: Glucose is a simple sugar and the primary source of energy for the body's cells.

AI: Glucose is a simple sugar and the primary source of energy for the body's cells.
```

## Setup

### Prerequisites

- [Python 3.8+](https://www.python.org/)
- [Ollama](https://ollama.com/) installed and running locally

### Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/langchain-react-agent.git
   cd langchain-react-agent
   ```

2. **Pull the Ollama model**
   ```bash
   ollama pull llama3.2:3b
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a `.env` file** in the project root with your API key:
   ```
   CURRENCY_API_KEY=your_api_key_here
   ```
   Get a free key at [exchangerate-api.com](https://www.exchangerate-api.com/)

6. **Run the agent**
   ```bash
   python main.py
   ```

## Project Structure

```
├── main.py             # LLM setup, ReAct prompt, agent executor, chat loop
├── tools.py            # Tool definitions (currency, translation, calculator)
├── config.py           # Loads API keys from .env
├── database.py         # Legacy file (not used in current version)
├── requirements.txt    # Python dependencies
├── .env                # API keys (not tracked by git)
├── .gitignore          # Files excluded from git
└── PROJECT_DOCUMENTATION.md  # Detailed project documentation
```

## Example Usage

```
You: what is 25 * 4?
AI: 25*4 = 100

You: translate good morning to French
AI: Translation: Bonjour

You: convert 50 EUR to GBP
AI: 50.0 EUR = 42.85 GBP

You: what is Python?
AI: Python is a high-level, interpreted programming language known for its simplicity.
```

## Tech Stack

- **LLM** — Ollama + llama3.2:3b (runs locally)
- **Framework** — LangChain (ReAct agent)
- **Prompt Strategy** — ReAct with few-shot examples + routing rules
- **APIs** — ExchangeRate-API, MyMemory Translation API
- **Language** — Python 3



