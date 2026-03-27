import json
import math
import requests
from langchain_core.tools import tool
import config


@tool
def convert_currency(query: str) -> str:
    """Use this tool ONLY when the user wants to convert or exchange money from one currency to another. Supports all major currencies like USD, INR, EUR, GBP, JPY, etc. Input must be a JSON string with three keys: amount (number), from_currency (currency code), to_currency (currency code). Example input: {"amount": 100, "from_currency": "USD", "to_currency": "INR"}"""
    try:
        params = json.loads(query)
        amount = float(params["amount"])
        from_curr = params["from_currency"].upper()
        to_curr = params["to_currency"].upper()

        url = f"https://v6.exchangerate-api.com/v6/{config.CURRENCY_API_KEY}/latest/{from_curr}"
        response = requests.get(url, timeout=10)
        data = response.json()
        rate = data["conversion_rates"].get(to_curr)
        if rate:
            result = amount * rate
            return f"{amount} {from_curr} = {result:.2f} {to_curr}"
        return f"Currency '{to_curr}' not found"
    except Exception as e:
        return f"Currency conversion failed: {e}"


@tool
def translate_text(query: str) -> str:
    """Use this tool ONLY when the user wants to translate English text into another language. Supports many languages. Input must be a JSON string with two keys: text (the English text to translate), to_language (language code). Common language codes: hi (Hindi), es (Spanish), fr (French), de (German), ja (Japanese), zh (Chinese), ar (Arabic), ko (Korean). Example input: {"text": "hello", "to_language": "hi"}"""
    try:
        params = json.loads(query)
        text = params["text"]
        to_lang = params["to_language"].lower()

        resp = requests.get("https://api.mymemory.translated.net/get",
                           params={"q": text, "langpair": f"en|{to_lang}"}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            translation = data["responseData"]["translatedText"]
            return f"Translation: {translation}"
        return f"Translation API error: {resp.status_code}"
    except Exception as e:
        return f"Translation failed: {e}"


@tool
def calculator(expression: str) -> str:
    """Use this tool ONLY when the user asks to calculate, compute, or solve a math expression. Input must be a plain math expression as a string. Supports: +, -, *, /, ** (power), sqrt(), abs(), round(), log(), sin(), cos(), tan(), pi, e. Example inputs: 2+2, 100/4, sqrt(16), 2**10, 3.14*5"""
    try:
        allowed = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round,
                   "pi": math.pi, "e": math.e, "log": math.log, "sin": math.sin,
                   "cos": math.cos, "tan": math.tan}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"
