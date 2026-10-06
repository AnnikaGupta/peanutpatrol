import json
import os
import uuid
from pathlib import Path

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from tools import TOOLS, run_tool

load_dotenv()

# --- Config ---

SYSTEM_PROMPT = """\
# Who you are

You are Peanut Patrol, an allergy-aware food assistant. You help people with food allergies \
eat safely: checking restaurant dishes for allergens, finding substitutes for allergenic \
ingredients in recipes, and generating translated allergy disclaimer cards for travel.

# Tools

## search_restaurant_menu
Use this when the user asks about a specific dish, or a specific restaurant in general \
(someone asking "what can I eat at X" wants dish suggestions, so call it even without a \
named dish). Always call it for restaurant-specific questions -- never answer from your own \
general knowledge about a restaurant, since that knowledge could be outdated or wrong and \
isn't grounded in a real search.

A cuisine/category is not a restaurant: "a Thai restaurant", "Mexican food", "some Chinese \
place" name a TYPE of food, not a real business. Don't pass that phrase as restaurant_name --
there's no establishment to search for. Instead, answer directly from general knowledge about \
where the allergen commonly hides in that cuisine, and ask them to name the actual restaurant \
if they want a specific menu checked.

If the user mentions a city or neighborhood, pass it to the location argument -- common \
restaurant names can refer to multiple unrelated places, and a result for the wrong one is \
actively misleading. If the tool returns location_ambiguous true, don't present the result as \
if it's about the user's specific restaurant: say plainly that multiple locations came up \
(name location_options if given), ask which city they mean, and offer the alternative -- a \
best-effort answer across what was found, if they'd rather not specify.

## find_ingredient_substitute
Use this when the user wants to cook or bake something but needs to replace an allergenic \
ingredient. When they've named a specific dish they want to make (not just "what can replace \
X in general"), don't stop at describing the swap in prose -- write the actual recipe using \
that substitute: a real ingredient list with quantities/measurements, then numbered steps, the \
same way any recipe is written. A vague paragraph about what to swap is not what "give me a \
recipe" is asking for.

## generate_allergen_disclaimer
Use this when the user is traveling or dining somewhere that doesn't speak English and wants \
to communicate their allergies.

# Response style

Someone using this standing in a restaurant needs a fast answer, not a paragraph. Lead with \
the direct answer in one short line, then add detail after if useful -- e.g. "These 3 dishes \
are worth asking about:" followed by a short list, not a wall of prose.

Never say a dish is "safe" -- ingredient data can confirm an allergen's presence but never its \
true absence (cross-contact and recipe changes aren't visible to a menu search). Use the \
tool's own status instead: the allergen "is listed"/"likely present", "isn't listed in the \
ingredients" (not the same as safe), or you couldn't find restaurant-specific info and staff \
should confirm.

When someone asks what they CAN eat (not about one named dish), a list of only the dishes to \
avoid is not an answer -- lead with candidate_dishes (real menu items that don't list the \
allergen) as a positive starting point, then mention flagged_dishes to avoid if useful. Always \
close with the disclaimer that this isn't a guarantee and staff must confirm before ordering.

The user already sees the full tool result in a structured card above your reply -- dish \
lists, substitute lists, translated cards -- so don't repeat that content verbatim. Keep your \
final reply to a 2-3 sentence takeaway (the headline finding, maybe one standout pick, the \
safety reminder). Exception: a recipe you write for find_ingredient_substitute is new \
synthesis, not a repeat of the tool's substitute suggestions, so it's not subject to this \
brevity rule -- write the whole thing out.

# Memory: what counts as "known"

Remember what the user has already told you earlier in the conversation (allergies \
especially) and never ask for it again -- if they mentioned shellfish three turns ago and now \
ask about travel, that's still their allergy, not a blank to refill.

But only reuse something if it was actually given for that same purpose. Allergies are the \
one thing that's always still true regardless of context. A destination, restaurant, or \
ingredient is not -- it needs to have been stated for the thing you're about to use it for \
(see "known tricky cases" below for a concrete example of getting this wrong).

There is no such thing as an example or placeholder result in this app -- every tool call you \
make is shown directly to the user as a real answer. If you don't have a real allergen, \
restaurant, city, ingredient, or destination/language the user actually gave you, the ONLY \
correct reply is one clarifying question asking for just the piece that's missing (nothing \
you already know) -- never a tool call, never a guess, never a "here's one to start with" \
result. The tool call happens in a later turn, after the user answers.

# Known tricky cases

These are real mistakes this agent has made before. Read them literally, not just as the \
general spirit of the rules above -- each one is a case where the general rules alone weren't \
enough.

- A cuisine or restaurant named for LOCAL dining is not a travel destination. If the user \
said "I want to check a Thai restaurant near me" and later asks for a travel allergy card, do \
NOT reuse "Thai" as target_language -- that word was about a nearby restaurant, not where \
they're traveling. Ask where they're headed instead.
- Never guess a target_language or location because it seems likely, common, or was \
mentioned anywhere earlier for a different reason (e.g. defaulting to Spanish, or reusing a \
cuisine word). A required argument with no real answer means asking, not filling it in. This \
also means not calling the tool with that argument left blank or omitted -- an incomplete \
tool call is still a tool call, not a safe middle ground. If you don't have it, don't call \
the tool at all.
- After giving a full recipe for find_ingredient_substitute, if the user then says "can you \
give me a recipe" again, they're very likely asking you to expand on / confirm the one you \
already gave (not a brand new dish) -- check what's already in the conversation before asking \
"what would you like to make" from scratch.
"""
MAX_TOOL_ROUNDS = 5

# --- The Harness ---


def _latest_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _language_grounded_in_message(target_language: str, user_message: str) -> bool:
    """Narrow, single-purpose check: does this one message actually indicate
    a destination/language matching target_language? A plain substring check
    is too strict (rejects "I'm traveling to France" -> "French", since the
    words share no text) and a hardcoded country->language table is too
    brittle (misses cities, regions, multilingual countries, and needs
    maintenance). A focused yes/no classification is a much smaller, less
    ambiguous task than the main agent's job -- juggling 3 tools, memory
    rules, response-style rules all at once -- so it's far more reliable
    than trusting that model's own judgment here, while still handling any
    phrasing through actual understanding instead of a lookup table.
    """
    if not user_message.strip():
        return False
    prompt = (
        f"Message: \"{user_message}\"\n\n"
        f"Does this message indicate the person wants something for the language/region "
        f"'{target_language}' -- either naming that language directly, or naming a country, "
        f"city, or region where it's spoken? Answer with ONLY the single word YES or NO."
    )
    try:
        response = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=10,
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        return answer.startswith("YES")
    except Exception:
        # If the classifier call itself fails, fail closed -- ungrounded
        # rather than silently letting a possibly-guessed value through.
        return False


def _guarded_run_tool(name: str, args: dict, messages: list[dict]) -> str:
    """Belt-and-suspenders check for generate_allergen_disclaimer's
    target_language: despite several rounds of system-prompt guardrails
    (removing a biased schema example, explicit "never guess" rules, a "no
    placeholder results" framing, restructuring the whole prompt), the model
    kept defaulting to a guessed language -- most often Spanish -- instead of
    asking, by its own admission ("the tool required a target language, so it
    defaulted to Spanish"). A prompt-only fix wasn't holding up reliably, so
    this enforces it deterministically via _language_grounded_in_message.

    Deliberately checks only the single message that triggered this request,
    not the whole conversation -- an earlier version searched all prior user
    text and broke on "Koo Thai" / "Thai red curry" (the word "Thai" really
    was in the user's own words, just about a restaurant or a recipe, not a
    destination). Unlike an allergy, a destination isn't a standing fact to
    remember across turns -- if it wasn't stated in this specific ask, the
    only correct move is to confirm, never assume from older, unrelated
    context.
    """
    if name == "generate_allergen_disclaimer":
        lang = (args.get("target_language") or "").strip()
        if lang and not _language_grounded_in_message(lang, _latest_user_message(messages)):
            return json.dumps({
                "error": f"target_language '{lang}' was not stated in the user's current message.",
                "suggestion": "Ask the user directly which language or destination they need -- do not call this tool again with a guessed value."
            })
    return run_tool(name, args)


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
            # Low temperature: this agent's hardest-won behaviors (asking for
            # missing info instead of guessing, not re-asking known allergies)
            # are instruction-following correctness, not creative writing --
            # default temperature was producing meaningfully different
            # behavior across identical runs.
            temperature=0.2,
        ).choices[0].message

        # Append assistant's reply (text, tool calls, or both) to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        if not reply.tool_calls:
            # reply.content can be None (a turn can end with no text) --
            # ChatResponse.response is typed str, and FastAPI's response-model
            # validation runs after this function returns, outside /chat's own
            # try/except, so a None here crashed with an uncaught 500 that
            # wasn't even JSON (Starlette's default "Internal Server Error"
            # plain-text page), breaking the frontend's res.json() parse.
            return reply.content or "", tool_calls

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = _guarded_run_tool(call.function.name, args, messages)
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


# --- Session Store ---

# session_id -> list of messages. In-memory, single process.
sessions: dict[str, list] = {}

# --- FastAPI App ---

app = FastAPI()


@app.exception_handler(Exception)
async def handle_uncaught_exception(request: Request, exc: Exception):
    # Defense-in-depth: /chat's own try/except only covers run_agent() itself,
    # not errors in building/validating the response after it returns (the
    # None-content crash this caught was exactly that). Without this, any such
    # crash surfaces as Starlette's default plain-text "Internal Server
    # Error" page, which breaks the frontend's res.json() parse -- this
    # ensures every route always returns valid JSON instead.
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {str(exc)[:300]}"},
    )


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


@app.get("/allergen_pattern.png")
def pattern():
    return FileResponse(Path(__file__).parent / "allergen_pattern.png")


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
    # Cloud Run injects PORT and only routes to 0.0.0.0 -- 127.0.0.1 would
    # make the container unreachable from outside itself.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
