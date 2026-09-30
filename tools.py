"""Peanut Patrol tools: restaurant menu search, ingredient substitutes, and allergy disclaimers."""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")

SPOONACULAR_BASE = "https://api.spoonacular.com"
ALLERGENS = [
    "peanuts", "tree nuts", "dairy", "milk", "eggs", "shellfish",
    "fish", "soy", "wheat", "gluten", "sesame", "sulfites", "mustard"
]

# ============================================================================
# Tool 1: search_restaurant_menu
# ============================================================================

def search_restaurant_menu(restaurant_name: str, dish_name: str, allergen: str) -> str:
    """Search for a dish at a restaurant and check allergen info.

    Args:
        restaurant_name: e.g., "Chipotle", "Panera", "Thai Palace"
        dish_name: e.g., "chicken bowl", "salad", "pad thai"
        allergen: e.g., "peanuts", "dairy", "gluten"

    Returns: JSON with dish info and allergen assessment
    """
    try:
        # Search recipes matching the restaurant + dish
        search_url = f"{SPOONACULAR_BASE}/recipes/search"
        params = {
            "query": f"{restaurant_name} {dish_name}",
            "number": 3,
            "apiKey": SPOONACULAR_API_KEY,
        }

        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code != 200:
            return json.dumps({
                "error": f"Restaurant API failed (status {response.status_code})",
                "suggestion": "Try asking the restaurant directly about allergens"
            })

        results = response.json().get("results", [])
        if not results:
            return json.dumps({
                "status": "inconclusive",
                "message": f"Couldn't find '{dish_name}' at {restaurant_name} online",
                "suggestion": "Contact the restaurant directly about allergen info"
            })

        # Get ingredient details for top result
        recipe_id = results[0]["id"]
        info_url = f"{SPOONACULAR_BASE}/recipes/{recipe_id}/information"
        info_response = requests.get(
            info_url,
            params={"apiKey": SPOONACULAR_API_KEY},
            timeout=10
        )

        if info_response.status_code != 200:
            return json.dumps({
                "status": "inconclusive",
                "message": "Found dish but couldn't load full details"
            })

        recipe = info_response.json()
        ingredients = recipe.get("extendedIngredients", [])

        # Check if allergen is in ingredients
        allergen_lower = allergen.lower()
        allergen_found = False
        allergen_ingredients = []

        for ing in ingredients:
            ing_name = ing.get("original", "").lower()
            if allergen_lower in ing_name:
                allergen_found = True
                allergen_ingredients.append(ing.get("original", ""))

        return json.dumps({
            "dish": results[0].get("title", "Unknown"),
            "restaurant": restaurant_name,
            "allergen_queried": allergen,
            "status": "contains" if allergen_found else "appears_safe",
            "allergen_ingredients": allergen_ingredients,
            "all_ingredients": [ing.get("original") for ing in ingredients[:5]],
            "warning": "⚠️ Always verify with restaurant—recipes may vary" if not allergen_found else "❌ This dish appears to contain your allergen"
        })

    except requests.RequestException as e:
        return json.dumps({
            "error": f"Network error: {str(e)[:100]}",
            "suggestion": "Check your internet connection or try again"
        })
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)[:100]}"})


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

TRANSLATIONS = {
    "spanish": {
        "base": "Tengo alergia a: {allergies}. Por favor, asegúrate de evitar la contaminación cruzada durante la preparación de alimentos.",
        "cultural": {
            "shellfish": "Muchos platos usan caldo de mariscos o salsa de camarones. Pregunta primero.",
            "peanuts": "La salsa de cacahuete es común en muchos platos. Verifica siempre.",
        }
    },
    "japanese": {
        "base": "私は{allergies}にアレルギーがあります。食事の準備時に交差汚染がないようにしてください。",
        "cultural": {
            "shellfish": "多くの日本料理は出汁（だし）を使用します。これはエビやカニを使います。確認してください。",
            "peanuts": "ピーナッツはアレルギー表示が必要な場合があります。事前に確認してください。",
        }
    },
    "french": {
        "base": "J'ai une allergie à: {allergies}. Veuillez vous assurer qu'il n'y a pas de contamination croisée lors de la préparation.",
        "cultural": {
            "dairy": "Le beurre et la crème sont courants dans la cuisine française. Demandez des alternatives.",
        }
    },
    "mandarin": {
        "base": "我对{allergies}过敏。请确保食物准备过程中没有交叉污染。",
        "cultural": {
            "shellfish": "许多中文菜肴使用虾酱或蚝油。准备前请确认。",
            "peanuts": "花生油在许多炒菜中使用。请询问。",
        }
    },
    "arabic": {
        "base": "أنا مصاب بحساسية من: {allergies}. يرجى التأكد من عدم التلوث المتبادل أثناء تحضير الطعام.",
        "cultural": {
            "shellfish": "العديد من الأطباق تستخدم منتجات بحرية. تأكد من المكونات.",
        }
    },
}


def generate_allergen_disclaimer(allergies: list, target_language: str) -> str:
    """Generate an allergy card in target language for travel/dining.

    Args:
        allergies: list of allergen names, e.g., ["peanuts", "dairy"]
        target_language: e.g., "Spanish", "Japanese", "French"

    Returns: JSON with translated disclaimer card
    """
    try:
        # Get translation
        language_lower = target_language.lower()
        allergie_list = ", ".join(allergies) if allergies else "unknown allergens"

        if language_lower in TRANSLATIONS:
            translated_text = TRANSLATIONS[language_lower]["base"].format(allergies=allergie_list)
            # Add cultural notes if available
            cultural_notes = []
            for allergen in allergies:
                allergen_lower = allergen.lower()
                if allergen_lower in TRANSLATIONS[language_lower]["cultural"]:
                    cultural_notes.append(TRANSLATIONS[language_lower]["cultural"][allergen_lower])
        else:
            # Fallback to English
            translated_text = f"I am allergic to: {allergie_list}. Please ensure no cross-contamination during food preparation."
            cultural_notes = []

        return json.dumps({
            "language": target_language,
            "translated_card": translated_text,
            "allergies": allergies,
            "cultural_tips": cultural_notes if cultural_notes else ["No specific tips for this language"],
            "usage": "Print this card or screenshot to show restaurants/chefs when traveling",
            "warning": "⚠️ Always show to staff before ordering to confirm safety",
            "supported_languages": list(TRANSLATIONS.keys())
        })

    except Exception as e:
        return json.dumps({
            "error": f"Disclaimer generation failed: {str(e)[:100]}",
            "fallback": f"I am allergic to: {', '.join(allergies)}"
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
            "description": "Generate a printable allergy card in any language for travel or dining at non-English restaurants. Great for communicating allergies when you don't speak the local language.",
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
                        "description": "Language for the card. Supported: Spanish, Japanese, French, Mandarin, Arabic. Others return English."
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
