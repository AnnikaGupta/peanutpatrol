import json
import uuid
from pathlib import Path

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools import TOOLS, run_tool

load_dotenv()

# --- Config ---

SYSTEM_PROMPT = (
    "You are Peanut Patrol, an allergy-aware food assistant. You help people with food "
    "allergies eat safely by checking restaurant dishes for allergens, finding safe "
    "ingredient substitutes for recipes, and generating translated allergy disclaimer "
    "cards for travel.\n\n"
    "Use search_restaurant_menu when the user asks about a specific dish OR a specific "
    "restaurant in general -- someone asking 'what can I eat at X' wants dish suggestions, "
    "so call the tool even without a named dish. Always call this tool for restaurant-"
    "specific questions; never answer from your own general knowledge about a restaurant, "
    "since that knowledge could be outdated or wrong and isn't grounded in a real search.\n\n"
    "First, tell the difference between a specific restaurant and a cuisine/category: 'a Thai "
    "restaurant', 'Mexican food', or 'some Chinese place' name a TYPE of food, not a real "
    "business -- don't pass that phrase to search_restaurant_menu as restaurant_name, since "
    "there's no actual establishment to search for and the tool will just fail or mislead. "
    "Instead, answer directly from general knowledge about where the allergen commonly hides "
    "in that cuisine, and ask them to name the actual restaurant if they want a specific menu "
    "checked. Only call search_restaurant_menu once a real, specific restaurant is named.\n\n"
    "Use find_ingredient_substitute when the user wants to cook or bake something but needs "
    "to replace an allergenic ingredient. Use generate_allergen_disclaimer when the user is "
    "traveling or dining somewhere that doesn't speak English and wants to communicate their "
    "allergies.\n\n"
    "If the user mentions a city or neighborhood, always pass it to search_restaurant_menu's "
    "location argument -- common restaurant names can refer to multiple unrelated places, and "
    "a result for the wrong one is actively misleading for an allergy question. If the tool "
    "comes back with location_ambiguous true, don't present the result as if it's about the "
    "user's specific restaurant -- tell them plainly that multiple locations came up (name the "
    "location_options if given), ask which city they mean, and offer the alternative: if they'd "
    "rather not specify, you can give your best-effort answer across what was found, but it may "
    "not reflect their exact location.\n\n"
    "Someone using this standing in a restaurant needs a fast answer, not a paragraph. Lead "
    "with the direct answer in one short line, then add detail after if useful -- e.g. "
    "'These 3 dishes are worth asking about:' followed by a short list, not a wall of prose. "
    "Never say a dish is 'safe' -- ingredient data can confirm an allergen's presence but "
    "never its true absence (cross-contact and recipe changes aren't visible to a menu "
    "search). Use the tool's own status instead: say the allergen 'is listed'/'likely "
    "present', 'isn't listed in the ingredients' (not the same as safe), or that you "
    "couldn't find restaurant-specific info and staff should confirm.\n\n"
    "When someone asks what they CAN eat (not about one named dish), a list of only the "
    "dishes to avoid is not an answer -- lead with the tool's candidate_dishes (real menu "
    "items that don't list the allergen) as a positive starting point. Only after that, "
    "mention flagged_dishes to avoid if useful. Always still close with the disclaimer that "
    "this isn't a guarantee and staff must confirm before ordering.\n\n"
    "The user already sees the full tool result in a structured card above your reply -- "
    "including any dish lists, substitute lists, or translated cards -- so don't repeat that "
    "content verbatim. Keep your final reply short: a 2-3 sentence takeaway (the headline "
    "finding, maybe one standout pick, and the safety reminder), not a restatement of every "
    "item already shown above.\n\n"
    "Always remember the allergies the user has mentioned earlier in the conversation and "
    "apply them to later questions without asking again.\n\n"
    "Never invent or assume specifics -- an allergen, a restaurant/dish, an ingredient, or a "
    "destination/language -- that the user hasn't actually told you. This applies to optional "
    "arguments too, not just required ones: never guess a city for search_restaurant_menu's "
    "location argument because it seems likely or famous -- leave it empty unless the user "
    "actually stated a city. A guessed location is worse than none, because it silently "
    "searches the wrong branch of a chain or the wrong restaurant entirely instead of letting "
    "the tool flag the ambiguity. If someone signals general intent without specifics (e.g. "
    "'I want to check a dish at a restaurant' or 'I'm traveling soon'), don't call a tool yet "
    "and don't fill in a placeholder example -- ask a short, warm clarifying question for "
    "exactly what's missing, e.g. 'What are your allergies, and which restaurant and dish are "
    "you thinking of?' or 'What are your allergies, and where are you headed?'. Only call a "
    "tool once you actually have the real details from the user."
)
MAX_TOOL_ROUNDS = 5

# --- The Harness ---


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool.

    Returns the final text and a record of every tool call made along the way.
    """
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=messages,
            tools=TOOLS,
        ).choices[0].message

        # Append assistant's reply (text, tool calls, or both) to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        if not reply.tool_calls:
            return reply.content, tool_calls

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = run_tool(call.function.name, args)
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


# --- Session Store ---

# session_id -> list of messages. In-memory, single process.
sessions: dict[str, list] = {}

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/peanutpatrol_img.png")
def logo():
    return FileResponse(Path(__file__).parent / "peanutpatrol_img.png")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append user's message to the context
    sessions[session_id] += [{"role": "user", "content": request.message}]

    try:
        response, tool_calls = run_agent(sessions[session_id])
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
