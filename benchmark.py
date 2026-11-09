import json
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types
import time
import os

load_dotenv()

openai_client = OpenAI()
claude_client = anthropic.Anthropic()
gemini_client = genai.Client()

MENU = {
    "Espresso": {"name": "Espresso", "category": "Coffee", "price": 2.50, "vegan": True, "allergens": [], "description": "Short, strong shot of coffee."},
    "Americano": {"name": "Americano", "category": "Coffee", "price": 3.00, "vegan": True, "allergens": [], "description": "Espresso diluted with hot water."},
    "Cappuccino": {"name": "Cappuccino", "category": "Coffee", "price": 3.80, "vegan": False, "allergens": ["milk"], "description": "Espresso with steamed milk and thick foam."},
    "Latte": {"name": "Latte", "category": "Coffee", "price": 4.20, "vegan": False, "allergens": ["milk"], "description": "Espresso with a lot of steamed milk."},
    "Flat White": {"name": "Flat White", "category": "Coffee", "price": 4.00, "vegan": False, "allergens": ["milk"], "description": "Espresso with velvety microfoam milk."},

    "Green Tea": {"name": "Green Tea", "category": "Tea", "price": 2.90, "vegan": True, "allergens": [], "description": "Light and fresh loose-leaf green tea."},
    "Earl Grey": {"name": "Earl Grey", "category": "Tea", "price": 2.90, "vegan": True, "allergens": [], "description": "Black tea with bergamot."},
    "Chai Latte": {"name": "Chai Latte", "category": "Tea", "price": 4.20, "vegan": False, "allergens": ["milk"], "description": "Spiced black tea with steamed milk."},

    "Iced Latte": {"name": "Iced Latte", "category": "Cold Drinks", "price": 4.50, "vegan": False, "allergens": ["milk"], "description": "Espresso, cold milk and ice."},
    "Cold Brew": {"name": "Cold Brew", "category": "Cold Drinks", "price": 3.90, "vegan": True, "allergens": [], "description": "Coffee steeped in cold water for 18 hours."},

    "Croissant": {"name": "Croissant", "category": "Pastry", "price": 3.10, "vegan": False, "allergens": ["gluten", "milk", "eggs"], "description": "Buttery, flaky French croissant."},
    "Blueberry Muffin": {"name": "Blueberry Muffin", "category": "Pastry", "price": 3.40, "vegan": True, "allergens": ["gluten"], "description": "Soft muffin with fresh blueberries."},
    "Almond Cookie": {"name": "Almond Cookie", "category": "Pastry", "price": 2.80, "vegan": True, "allergens": ["gluten", "nuts"], "description": "Chewy cookie with almond pieces."},
}

TEST_CASES = [
    ("how much is a latte?", [{"items": ["Latte"]}]),
    ("what teas do you have?", [{"category": "Tea"}]),
    ("tell me about the kappuccino and the croissant", [{"items": ["Cappuccino", "Croissant"]}]),
    ("how much is a koffee lattee?", [{"items": ["Latte"]}]),
    ("is the FAT WHITEE vegan?", [{"items": ["Flat White"]}]),
    ("what do you have in cold drinks?", [{"category": "Cold Drinks"}]),
    ("I want a latte and some pastries", [{"items": ["Latte"]}, {"category": "Pastry"}]),
    ("write me a poem", []),
    ("do you have burgers?", []),
    ("what's the cheapest drink?", []),
]

DELIMITER = "####"
CATEGORIES = sorted({item["category"] for item in MENU.values()})
ITEM_NAMES = list(MENU.keys())

EXTRACTION_PROMPT = f"""\
You will be provided with customer queries for a coffee shop. The customer query will be delimited with {DELIMITER} characters.

Output a JSON list of objects. Each object must have exactly one of the following keys:
    "items": a list of menu item names mentioned in the query
    "category": one of the allowed categories, used when the customer asks about a whole category and names no specific item

Use only double quotes so the output is valid JSON.
Item names and categories must come from the allowed lists below. If the customer writes a name in a different spelling or case, output the exact name from the list.
Customers may make typos or use informal names (for example "lattee", "flat wite", "coffee latte"). If a word is clearly a misspelling or variant of a name from the allowed list, output the exact name from the list. If the customer asks for something that is not on the menu at all (for example "burger" or "pizza"), do not map it to the closest item. In that case output an empty list: []
If the query does not mention any menu items or categories, output an empty list: []

Allowed categories: {", ".join(CATEGORIES)}
Allowed items: {", ".join(ITEM_NAMES)}

Examples:
Query: {DELIMITER}How much is a cappuccino, and what teas do you have?{DELIMITER}
Output: [{{"items": ["Cappuccino"]}}, {{"category": "Tea"}}]

Query: {DELIMITER}how much is a lattee and a flat wite?{DELIMITER}
Output: [{{"items": ["Latte", "Flat White"]}}]

Only output the JSON list, with nothing else.
"""

def call_openai(system, user):
    start = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-5.4-mini",
        max_completion_tokens=300,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return {
        "text": response.choices[0].message.content,
        "latency": time.time() - start,
        "tokens": response.usage.total_tokens,
    }
    
def call_claude(system, user):
    start = time.time()
    response = claude_client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=300,
        system=system,                                   
        messages=[{"role": "user", "content": user}],
    )
    return {
        "text": response.content[0].text,
        "latency": time.time() - start,
        "tokens": response.usage.input_tokens + response.usage.output_tokens,
    }
    
def call_gemini(system, user):
    start = time.time()
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=300
        )
    )
    return {
        "text": response.text,
        "latency": time.time() - start,
        "tokens": response.usage_metadata.total_token_count,
    }

def parse_extraction(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("Error: model did not return valid JSON")
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]

def main():
    question = "how much is a latte?"
    user = f"{DELIMITER}{question}{DELIMITER}"

    PROVIDERS = {
        "openai": call_openai,
        "claude": call_claude,
        "gemini": call_gemini,
    }   

    for provider in PROVIDERS:
        result = PROVIDERS[provider](EXTRACTION_PROMPT, user)
        print(provider, result, "\n")
if __name__ == "__main__":
    main()