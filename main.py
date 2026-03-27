from langchain_ollama import OllamaLLM
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from tools import convert_currency, translate_text, calculator

# Step 1: Create the LLM (text-based for ReAct)
llm = OllamaLLM(model="llama3.2:3b", temperature=0)

# Step 2: ReAct prompt with strict routing rules and few-shot examples
react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

TOOL ROUTING RULES:
- USE convert_currency ONLY for converting/exchanging money between currencies
- USE translate_text ONLY for translating text to another language
- USE calculator ONLY for math calculations or arithmetic
- DO NOT use any tool for general knowledge, greetings, opinions, or definitions. Give Final Answer directly.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 2 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

EXAMPLES:

Question: What is glucose?
Thought: This is a general knowledge question. I can answer directly without any tool.
Final Answer: Glucose is a simple sugar and the primary source of energy for the body's cells.

Question: What is 5 + 3?
Thought: The user wants to do math. I should use calculator.
Action: calculator
Action Input: 5+3
Observation: 5+3 = 8
Thought: I now know the final answer
Final Answer: 5 + 3 = 8

Question: Convert 50 USD to EUR
Thought: The user wants to convert currency. I should use convert_currency.
Action: convert_currency
Action Input: {{"amount": 50, "from_currency": "USD", "to_currency": "EUR"}}

Question: Translate goodbye to Spanish
Thought: The user wants to translate text. I should use translate_text.
Action: translate_text
Action Input: {{"text": "goodbye", "to_language": "es"}}

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# Step 3: Create the agent and executor
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

# Step 4: Handle the response
def ask_agent(user_input):
    response = executor.invoke({"input": user_input})

    # Use tool output directly if available
    for step in response.get("intermediate_steps", []):
        tool_output = step[1]
        if tool_output and "error" not in tool_output.lower():
            return tool_output

    return response["output"]

# Step 5: Chat loop
print("LangChain ReAct Agent Ready!")
print("Examples: 'Convert 100 USD to INR', 'Translate hello to Hindi', 'What is 25 * 4?'")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    if not user_input:
        continue

    answer = ask_agent(user_input)
    print(f"\nAI: {answer}\n")
