"""Peanut Patrol tools: restaurant menu search, ingredient substitutes, and allergy disclaimers."""

import json
import os
import requests
import litellm
from dotenv import load_dotenv

load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")
# GOOGLE_CLOUD_PROJECT is read automatically by google-auth / litellm's Vertex
# integration from the environment -- no need to pass it explicitly per call.

SPOONACULAR_BASE = "https://api.spoonacular.com"
ALLERGENS = [
    "peanuts", "tree nuts", "dairy", "milk", "eggs", "shellfish",
    "fish", "soy", "wheat", "gluten", "sesame", "sulfites", "mustard"
]

# ============================================================================
# Tool 1: search_restaurant_menu
# ============================================================================
#
# Spoonacular's recipe database was tried first, but it's a home-cook recipe
# index, not a restaurant menu database: searching "Chipotle chicken bowl"
# matched an unrelated home recipe called "...Chipotle Dressing" (the pepper,
# not the chain), and would have reported its ingredients as if they were the
# real restaurant's. For a safety-critical allergen tool, a confidently wrong
# answer is worse than an honest "I don't know" -- so this instead uses a
# Gemini call with Google Search grounding (same Vertex credentials as the
# main agent, no extra API key) to find the restaurant's actual posted
# allergen info when it exists online, and otherwise gives general dish-level
# knowledge with an explicit "not restaurant-verified" caveat. This also
# covers small independent restaurants (e.g. a specific NYC spot) that would
# never appear in any recipe or chain database.

def search_restaurant_menu(restaurant_name: str, dish_name: str, allergen: str) -> str:
    """Check whether a dish at a specific restaurant likely contains an allergen.

    Args:
        restaurant_name: e.g., "Chipotle", "Panera", "Koo Thai"
        dish_name: e.g., "chicken bowl", "salad", "drunken noodles"
        allergen: e.g., "peanuts", "dairy", "gluten"

    Returns: JSON with findings and an explicit confidence/source label
    """
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        return json.dumps({
            "error": "GOOGLE_CLOUD_PROJECT not set in .env",
            "suggestion": "Contact the restaurant directly about allergen info"
        })

    prompt = (
        f"Someone with a {allergen} allergy wants to know if '{dish_name}' at "
        f"'{restaurant_name}' is safe to eat. Search for this specific restaurant's "
        f"posted menu, ingredient list, or allergen guide if it exists online. "
        f"If you find restaurant-specific information, say so clearly and summarize it. "
        f"If you cannot find anything specific to this restaurant, say that explicitly, "
        f"then give general knowledge about whether {allergen} is typically found in this "
        f"type of dish. Always end by recommending the user confirm with restaurant staff "
        f"directly, since recipes and cross-contamination risk vary by location. "
        f"Keep the answer under 150 words."
    )

    try:
        response = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=[{"role": "user", "content": prompt}],
            tools=[{"googleSearch": {}}],
            timeout=20,
        )
        findings = response.choices[0].message.content
        if not findings:
            return json.dumps({
                "status": "inconclusive",
                "message": f"No information found for '{dish_name}' at {restaurant_name}",
                "suggestion": "Contact the restaurant directly about allergen info"
            })

        return json.dumps({
            "restaurant": restaurant_name,
            "dish": dish_name,
            "allergen_queried": allergen,
            "findings": findings,
            "source": "gemini_web_search",
            "warning": "⚠️ AI-summarized web search result, not verified restaurant data. Always confirm with staff before ordering."
        })

    except Exception as e:
        return json.dumps({
            "error": f"Search failed: {str(e)[:150]}",
            "suggestion": "Contact the restaurant directly about allergen info"
        })


# ============================================================================
# Tool 2: find_ingredient_substitute
# ============================================================================

def find_ingredient_substitute(allergenic_ingredient: str, recipe_context: str = "") -> str:
    """Find safe substitutes for an allergenic ingredient in a recipe.

    Args:
        allergenic_ingredient: e.g., "peanut butter", "butter", "eggs"
        recipe_context: optional, e.g., "cookies", "sauce"

    Returns: JSON with substitution suggestions
    """
    try:
        sub_url = f"{SPOONACULAR_BASE}/food/ingredients/substitutes"
        params = {
            "ingredientName": allergenic_ingredient,
            "apiKey": SPOONACULAR_API_KEY,
        }

        response = requests.get(sub_url, params=params, timeout=10)
        api_substitutes = []
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                api_substitutes = data.get("substitutes", [])

        if api_substitutes:
            return json.dumps({
                "ingredient": allergenic_ingredient,
                "recipe_type": recipe_context or "general",
                "substitutes": api_substitutes,
                "source": "Spoonacular",
                "note": "Ratios shown as '<original amount> = <substitute amount>'"
            })

        # API had no data for this ingredient (common for allergen-specific terms
        # like "shellfish" or "eggs") -- fall back to curated allergy substitutes.
        fallback = get_common_substitutes(allergenic_ingredient)
        return json.dumps({
            "ingredient": allergenic_ingredient,
            "recipe_type": recipe_context or "general",
            "substitutes": fallback,
            "source": "curated_fallback",
            "note": "Spoonacular had no data for this ingredient; using allergy-safe substitutes instead"
        })

    except requests.RequestException as e:
        return json.dumps({
            "ingredient": allergenic_ingredient,
            "substitutes": get_common_substitutes(allergenic_ingredient),
            "source": "curated_fallback",
            "note": f"Network error reaching Spoonacular ({str(e)[:60]}); using allergy-safe substitutes instead"
        })
    except Exception as e:
        return json.dumps({"error": f"Error finding substitutes: {str(e)[:100]}"})


def get_common_substitutes(ingredient: str) -> list:
    """Return common allergen-free substitutes for an ingredient."""
    ingredient_lower = ingredient.lower()

    substitutes = {
        "peanut butter": [
            {"substitute": "sunflower seed butter", "ratio": "1:1", "reason": "similar creamy texture and taste"},
            {"substitute": "tahini (sesame)", "ratio": "1:1", "reason": "nutty flavor, good protein"},
            {"substitute": "almond butter (tree nut)", "ratio": "1:1", "reason": "if tree nuts OK"},
        ],
        "butter": [
            {"substitute": "coconut oil", "ratio": "1:1", "reason": "works in most baking"},
            {"substitute": "applesauce", "ratio": "1:1", "reason": "for cakes and brownies, adds moisture"},
            {"substitute": "olive oil", "ratio": "0.75:1", "reason": "good for savory dishes"},
        ],
        "eggs": [
            {"substitute": "applesauce", "ratio": "0.25 cup per egg", "reason": "binder in baking"},
            {"substitute": "aquafaba (chickpea liquid)", "ratio": "3 tbsp per egg", "reason": "great for vegan baking"},
            {"substitute": "flax egg (flax + water)", "ratio": "1 tbsp flax + 3 tbsp water", "reason": "works as binder"},
        ],
        "dairy milk": [
            {"substitute": "oat milk", "ratio": "1:1", "reason": "creamy, neutral flavor"},
            {"substitute": "almond milk", "ratio": "1:1", "reason": "lighter, works in coffee/cereal"},
            {"substitute": "coconut milk", "ratio": "1:1", "reason": "richer taste"},
        ],
        "shellfish": [
            {"substitute": "mushrooms (umami flavor)", "ratio": "same amount", "reason": "replicates savory depth"},
            {"substitute": "white fish (if fish OK)", "ratio": "1:1", "reason": "similar texture"},
        ],
    }

    # Match ingredient (partial match for flexibility)
    for key, subs in substitutes.items():
        if ingredient_lower in key or key in ingredient_lower:
            return subs

    # Fallback
    return [{"substitute": "Ask a chef or recipe site for alternatives", "reason": "No common substitute found"}]


# ============================================================================
# Tool 3: generate_allergen_disclaimer
# ============================================================================
#
# Originally a hardcoded dict of 5 languages with cultural notes I wrote from
# memory (and got Spanish/Mexican cuisine wrong on: mole and peanut oil were
# missing entirely). Replaced with a single grounded Gemini call, same pattern
# as search_restaurant_menu: works for any language (not just 5), and the
# cultural notes come from a web-grounded lookup instead of guesswork.

def generate_allergen_disclaimer(allergies: list, target_language: str) -> str:
    """Generate a printable allergy card in any language, with cuisine-specific
    dishes/ingredients to watch for.

    Args:
        allergies: list of allergen names, e.g., ["peanuts", "dairy"]
        target_language: any language name, e.g., "Spanish", "Thai", "Swahili"

    Returns: JSON with translated card text and cuisine-specific cultural notes
    """
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        return json.dumps({"error": "GOOGLE_CLOUD_PROJECT not set in .env"})

    allergy_list = ", ".join(allergies) if allergies else "unknown allergens"
    prompt = (
        f"A traveler is allergic to: {allergy_list}. Help them prepare for dining out "
        f"in a region where {target_language} is spoken.\n\n"
        f"1. Translate this into {target_language}, polite and clear for restaurant staff: "
        f"\"I am allergic to {allergy_list}. Please ensure no cross-contamination during "
        f"food preparation.\"\n"
        f"2. Identify the cuisine(s) most associated with {target_language}-speaking regions. "
        f"Search the web if it helps you be accurate. List 2-4 SPECIFIC real dishes, sauces, or "
        f"cooking practices from that cuisine that commonly contain {allergy_list}, so the "
        f"traveler knows exactly what to ask about. Name actual dishes/sauces, not generic advice.\n\n"
        f"Respond with ONLY valid JSON, no markdown fences, in this exact shape:\n"
        f'{{"translated_card": "<translated text>", "cultural_notes": ["<note 1>", "<note 2>"]}}'
    )

    try:
        response = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=[{"role": "user", "content": prompt}],
            tools=[{"googleSearch": {}}],
            timeout=20,
        )
        raw = (response.choices[0].message.content or "").strip()
        # Models sometimes wrap JSON in markdown fences despite instructions.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return json.dumps({
                "language": target_language,
                "allergies": allergies,
                "translated_card": raw,
                "cultural_notes": [],
                "note": "Model did not return structured JSON; showing raw response.",
                "usage": "Print this card or screenshot to show restaurants/chefs when traveling",
                "warning": "⚠️ Always show to staff before ordering to confirm safety"
            })

        return json.dumps({
            "language": target_language,
            "allergies": allergies,
            "translated_card": parsed.get("translated_card", ""),
            "cultural_notes": parsed.get("cultural_notes", []),
            "source": "gemini_web_search",
            "usage": "Print this card or screenshot to show restaurants/chefs when traveling",
            "warning": "⚠️ Always show to staff before ordering to confirm safety"
        })

    except Exception as e:
        return json.dumps({
            "error": f"Disclaimer generation failed: {str(e)[:100]}",
            "fallback": f"I am allergic to: {allergy_list}"
        })


# ============================================================================
# Tool Registry (what Gemini sees)
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_restaurant_menu",
            "description": "Search for a dish at a restaurant and check if it contains specific allergens. Helps you find safe food options when dining out.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_name": {
                        "type": "string",
                        "description": "Name of the restaurant, e.g., 'Chipotle', 'Panera', 'Thai Palace'"
                    },
                    "dish_name": {
                        "type": "string",
                        "description": "Name of the dish to check, e.g., 'chicken bowl', 'salad', 'pad thai'"
                    },
                    "allergen": {
                        "type": "string",
                        "description": f"Allergen to check for. Options: {', '.join(ALLERGENS)}"
                    },
                },
                "required": ["restaurant_name", "dish_name", "allergen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_ingredient_substitute",
            "description": "Find safe alternatives to replace an allergenic ingredient in a recipe. Helps you adapt recipes you love to your allergies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "allergenic_ingredient": {
                        "type": "string",
                        "description": "The ingredient you want to replace, e.g., 'peanut butter', 'butter', 'eggs', 'dairy milk'"
                    },
                    "recipe_context": {
                        "type": "string",
                        "description": "Optional: what you're making (e.g., 'cookies', 'sauce', 'cake'). Helps give better suggestions."
                    },
                },
                "required": ["allergenic_ingredient"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_allergen_disclaimer",
            "description": "Generate a printable allergy card in any language for travel or dining at non-English restaurants, plus specific dishes/sauces from that region's cuisine to watch out for. Great for communicating allergies when you don't speak the local language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "allergies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"List of allergens. Examples: {', '.join(ALLERGENS[:5])}"
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Any language name, e.g. 'Spanish', 'Thai', 'Swahili'."
                    },
                },
                "required": ["allergies", "target_language"],
            },
        },
    },
]

# Map tool names to functions
TOOL_MAP = {
    "search_restaurant_menu": search_restaurant_menu,
    "find_ingredient_substitute": find_ingredient_substitute,
    "generate_allergen_disclaimer": generate_allergen_disclaimer,
}


def run_tool(name: str, args: dict) -> str:
    """Execute a tool. Models invent tool names; never let that crash the loop."""
    if name not in TOOL_MAP:
        return json.dumps({
            "error": f"Unknown tool '{name}'. Available: {list(TOOL_MAP.keys())}"
        })
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return json.dumps({
            "error": f"Bad arguments for {name}: {str(e)[:100]}"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Tool execution failed: {str(e)[:100]}"
        })
