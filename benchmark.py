import json
import time

from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types

load_dotenv()

openai_client = OpenAI()
claude_client = anthropic.Anthropic()
gemini_client = genai.Client()

JUDGE_PROVIDER = "openai"
DELIMITER = "####"

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

    ("how much is the espresso?", [{"items": ["Espresso"]}]),
    ("is the cold brew vegan?", [{"items": ["Cold Brew"]}]),
    ("what's in a blueberry muffin?", [{"items": ["Blueberry Muffin"]}]),
    ("does the croissant contain gluten?", [{"items": ["Croissant"]}]),
    ("tell me about the earl grey", [{"items": ["Earl Grey"]}]),
    ("is the almond cookie vegan?", [{"items": ["Almond Cookie"]}]),
    ("what's in the iced latte?", [{"items": ["Iced Latte"]}]),

    ("how much is a capuccino?", [{"items": ["Cappuccino"]}]),
    ("do you have a croisant?", [{"items": ["Croissant"]}]),
    ("how much is green tee?", [{"items": ["Green Tea"]}]),
    ("is the chai lattte vegan?", [{"items": ["Chai Latte"]}]),
    ("i would like an espreso", [{"items": ["Espresso"]}]),
    ("what is an americanno?", [{"items": ["Americano"]}]),
    ("is the earl gray vegan?", [{"items": ["Earl Grey"]}]),
    ("how much is a blueberry mufin?", [{"items": ["Blueberry Muffin"]}]),
    ("does the almond cokie have nuts?", [{"items": ["Almond Cookie"]}]),
    ("price of iced late?", [{"items": ["Iced Latte"]}]),

    ("how much is the iced latte?", [{"items": ["Iced Latte"]}]),
    ("is the chai latte vegan?", [{"items": ["Chai Latte"]}]),
    ("what is the difference between a latte and a flat white?", [{"items": ["Latte", "Flat White"]}]),
    ("how much is a cold brew and an iced latte?", [{"items": ["Cold Brew", "Iced Latte"]}]),

    ("show me all the coffees", [{"category": "Coffee"}]),
    ("what pastries do you sell?", [{"category": "Pastry"}]),
    ("what kinds of tea are there?", [{"category": "Tea"}]),
    ("list your cold drinks please", [{"category": "Cold Drinks"}]),
    ("what coffee drinks do you have?", [{"category": "Coffee"}]),
    ("do you have any teas?", [{"category": "Tea"}]),

    ("i'd like an espresso, an americano and a croissant", [{"items": ["Espresso", "Americano", "Croissant"]}]),
    ("compare the cappuccino and the flat white", [{"items": ["Cappuccino", "Flat White"]}]),
    ("how much are the green tea and the earl grey?", [{"items": ["Green Tea", "Earl Grey"]}]),
    ("i want a blueberry muffin and an almond cookie", [{"items": ["Blueberry Muffin", "Almond Cookie"]}]),

    ("how much is a croissant, and what teas do you have?", [{"items": ["Croissant"]}, {"category": "Tea"}]),
    ("what cold drinks do you have, and is the espresso vegan?", [{"category": "Cold Drinks"}, {"items": ["Espresso"]}]),
    ("i want an americano and what pastries do you have?", [{"items": ["Americano"]}, {"category": "Pastry"}]),

    ("hi", []),
    ("what time do you open?", []),
    ("do you have a mocha?", []),
    ("do you have any pizza?", []),
    ("tell me a joke", []),
    ("where are you located?", []),
    ("do you have a smoothie?", []),
]

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

ANSWER_PROMPT = (
    "You are a friendly assistant for a coffee shop. Answer the customer's question "
    "using ONLY the menu information provided. Never guess or invent items, prices or "
    "ingredients. All prices are in US dollars, so write them like $4.20. "
    "If the menu information says 'No matching menu items found', it means "
    "the customer asked about something we do not sell, or the question is not about a "
    "specific menu item. Say that clearly and briefly (for example 'We don't have that') "
    "and, if it is a coffee shop question, offer to help with the items we do have. "
    "If the question is not about the menu at all, say you can only help with menu questions. "
    "Keep the answer short, in plain text."
)

JUDGE_PROMPT = f"""\
You are grading the answer of a coffee shop assistant. You get the customer's question and the assistant's answer. The question and the answer are delimited with {DELIMITER} characters. Use ONLY this menu as the source of truth:

{json.dumps(MENU, indent=2)}

Grade the answer PASS only if ALL of these are true:
1. Every price, vegan status, allergen and description it states matches the menu. Prices must be in US dollars or have no currency symbol. Any other currency symbol is a FAIL.
2. It does not mention any item that is not on the menu as if it were available.
3. It actually answers the question, including all items or the whole category asked about.
4. If the question is not about the menu, or asks for something not on the menu, the answer says so instead of inventing information.

Also:
- If the customer misspelled an item name and the answer gives correct information for the item the customer most likely meant, that is correct. The answer does not have to mention the misspelling.
- Short answers are fine. Do not fail an answer for missing extra details the customer did not ask for.

You are also told which menu entries are correct for the question. If that list is not empty, the answer FAILS if it says the item is not available or gives no information about those entries. If the list is empty, the question is not about anything we sell, and the answer must say so.

Respond with only a JSON object: {{"verdict": "PASS" or "FAIL", "reason": "<one short sentence>"}}
"""


def call_openai(system, user):
    start = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-5.4-mini",
        max_completion_tokens=1000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return {
        "text": response.choices[0].message.content or "",
        "latency": time.time() - start,
        "tokens": response.usage.total_tokens,
    }


def call_claude(system, user):
    start = time.time()
    response = claude_client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return {
        "text": text,
        "latency": time.time() - start,
        "tokens": response.usage.input_tokens + response.usage.output_tokens,
    }


def call_gemini(system, user):
    for attempt in range(5):
        start = time.time()
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=1000,
                ),
            )
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                time.sleep(15)
                continue
            raise
        latency = time.time() - start
        time.sleep(4.5)
        return {
            "text": response.text or "",
            "latency": latency,
            "tokens": response.usage_metadata.total_token_count,
        }


PROVIDERS = {
    "openai": call_openai,
    "claude": call_claude,
    "gemini": call_gemini,
}


def parse_extraction(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def lookup(parsed):
    results = []
    for entry in parsed:
        if "items" in entry:
            for name in entry["items"]:
                item = MENU.get(name)
                if item and item not in results:
                    results.append(item)
        elif "category" in entry:
            for item in MENU.values():
                if item["category"] == entry["category"] and item not in results:
                    results.append(item)
    return results


def answer_question(provider, question):
    call = PROVIDERS[provider]
    user = f"{DELIMITER}{question}{DELIMITER}"

    extraction = call(EXTRACTION_PROMPT, user)
    parsed = parse_extraction(extraction["text"])
    results = lookup(parsed)
    if results:
        menu_info = json.dumps(results, indent=2)
    else:
        menu_info = "No matching menu items found."

    answer = call(ANSWER_PROMPT, f"Menu information:\n{menu_info}\n\nCustomer question: {question}")
    return {
        "parsed": parsed,
        "text": answer["text"],
        "latency": extraction["latency"] + answer["latency"],
        "tokens": extraction["tokens"] + answer["tokens"],
    }


def judge_answer(question, answer, expected):
    user = (
        f"Question: {DELIMITER}{question}{DELIMITER}\n"
        f"Answer: {DELIMITER}{answer}{DELIMITER}\n"
        f"Correct menu entries for this question: {json.dumps(expected)}"
    )
    try:
        text = PROVIDERS[JUDGE_PROVIDER](JUDGE_PROMPT, user)["text"]
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        return data["verdict"].upper() == "PASS", data["reason"]
    except Exception as e:
        return None, f"judge error: {e}"


def main():
    print("Judge:", JUDGE_PROVIDER)

    for provider in PROVIDERS:
        print(f"\n=== {provider} ===")
        extraction_score = 0
        answer_score = 0
        judged = 0
        errors = 0
        total_time = 0
        total_tokens = 0

        for question, expected in TEST_CASES:
            try:
                result = answer_question(provider, question)
            except Exception as e:
                print(f"ERROR: {question} | {e}")
                errors += 1
                continue

            answer_ok, reason = judge_answer(question, result["text"], expected)
            extraction_ok = result["parsed"] == expected

            extraction_score += extraction_ok
            if answer_ok is not None:
                judged += 1
                answer_score += answer_ok
            total_time += result["latency"]
            total_tokens += result["tokens"]

            if not extraction_ok or answer_ok is not True:
                print(f"Q: {question}")
                print(f"  extraction: {'OK' if extraction_ok else 'FAIL'} | expected {expected} | got {result['parsed']}")
                print(f"  answer: {answer_ok} | {reason}")
                print(f"  text: {result['text']}")

        done = len(TEST_CASES) - errors
        print(f"\n{provider}: extraction {extraction_score}/{done}, answers {answer_score}/{judged}, errors {errors}")
        if done:
            print(f"{provider}: avg time {total_time / done:.2f}s, tokens {total_tokens}")


if __name__ == "__main__":
    main()