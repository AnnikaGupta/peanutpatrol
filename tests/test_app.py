"""Tests for app.py's FastAPI endpoints and the run_agent harness loop.
litellm.completion is mocked throughout -- these test the harness's control
flow (sessions, tool-call looping, error handling), not real model behavior.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture(autouse=True)
def clear_sessions():
    """The session store is a module-level dict -- reset it between tests
    so one test's session doesn't leak into another's."""
    app_module.sessions.clear()
    yield
    app_module.sessions.clear()


@pytest.fixture
def client():
    return TestClient(app_module.app)


def make_message(content=None, tool_calls=None):
    """A fake litellm reply message with the attributes run_agent reads."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {
        "role": "assistant", "content": content,
        "tool_calls": tool_calls,
    }
    return msg


def make_completion(message):
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    return resp


def make_tool_call(call_id, name, arguments: dict):
    call = MagicMock()
    call.id = call_id
    call.function.name = name
    call.function.arguments = json.dumps(arguments)
    return call


# ============================================================================
# Static routes
# ============================================================================

class TestStaticRoutes:
    def test_index_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]

    def test_logo_served(self, client):
        res = client.get("/peanutpatrol_img.png")
        assert res.status_code == 200

    def test_pattern_served(self, client):
        res = client.get("/allergen_pattern.png")
        assert res.status_code == 200


# ============================================================================
# /chat: sessions
# ============================================================================

class TestChatSessions:
    def test_no_session_id_creates_new_one(self, client):
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="Hi there!")
        )):
            res = client.post("/chat", json={"message": "hello", "session_id": None})
        assert res.status_code == 200
        body = res.json()
        assert body["session_id"]  # non-empty
        assert body["response"] == "Hi there!"
        assert body["tool_calls"] == []

    def test_two_fresh_requests_get_different_sessions(self, client):
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="Hi!")
        )):
            r1 = client.post("/chat", json={"message": "hello", "session_id": None})
            r2 = client.post("/chat", json={"message": "hello", "session_id": None})
        assert r1.json()["session_id"] != r2.json()["session_id"]

    def test_existing_session_id_is_reused(self, client):
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="ok")
        )):
            r1 = client.post("/chat", json={"message": "first", "session_id": None})
            sid = r1.json()["session_id"]
            r2 = client.post("/chat", json={"message": "second", "session_id": sid})
        assert r2.json()["session_id"] == sid

    def test_conversation_history_accumulates_in_session(self, client):
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="ok")
        )):
            r1 = client.post("/chat", json={"message": "I am allergic to shellfish", "session_id": None})
            sid = r1.json()["session_id"]
            client.post("/chat", json={"message": "what did I just say?", "session_id": sid})

        messages = app_module.sessions[sid]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        assert "I am allergic to shellfish" in user_messages
        assert "what did I just say?" in user_messages

    def test_sessions_are_isolated(self, client):
        """A message sent in session A must never appear in session B's history."""
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="ok")
        )):
            rA = client.post("/chat", json={"message": "I am allergic to peanuts", "session_id": None})
            sidA = rA.json()["session_id"]
            rB = client.post("/chat", json={"message": "I am allergic to dairy", "session_id": None})
            sidB = rB.json()["session_id"]

        contents_a = [m["content"] for m in app_module.sessions[sidA]]
        contents_b = [m["content"] for m in app_module.sessions[sidB]]
        assert "I am allergic to dairy" not in contents_a
        assert "I am allergic to peanuts" not in contents_b

    def test_new_session_starts_with_system_prompt(self, client):
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="ok")
        )):
            r = client.post("/chat", json={"message": "hi", "session_id": None})
        sid = r.json()["session_id"]
        assert app_module.sessions[sid][0] == {"role": "system", "content": app_module.SYSTEM_PROMPT}


# ============================================================================
# /chat: tool-calling loop
# ============================================================================

class TestChatToolLoop:
    def test_single_tool_call_then_final_answer(self, client):
        tool_call = make_tool_call("call_1", "find_ingredient_substitute", {"allergenic_ingredient": "butter"})
        first_reply = make_message(content=None, tool_calls=[tool_call])
        second_reply = make_message(content="Use olive oil instead.", tool_calls=None)

        # find_ingredient_substitute is guarded (see TestGuardedTargetLanguage's
        # sibling tests for that logic specifically) -- a classifier call sits
        # between the two orchestrator calls, mocked here to say YES so this
        # test can focus on the general tool-call recording flow.
        with patch.object(app_module.litellm, "completion", side_effect=[
            make_completion(first_reply), make_classifier_response("YES"), make_completion(second_reply),
        ]):
            res = client.post("/chat", json={
                "message": "I'm allergic to butter, what's a substitute?", "session_id": None,
            })

        body = res.json()
        assert body["response"] == "Use olive oil instead."
        assert len(body["tool_calls"]) == 1
        assert body["tool_calls"][0]["name"] == "find_ingredient_substitute"
        assert body["tool_calls"][0]["args"] == {"allergenic_ingredient": "butter"}
        # The result should be the real run_tool output (real curated fallback,
        # since requests.get isn't mocked here) -- just confirm it's valid JSON
        # with the expected shape rather than a literal network call failure.
        parsed_result = json.loads(body["tool_calls"][0]["result"])
        assert "ingredient" in parsed_result or "error" in parsed_result

    def test_no_tool_call_returns_immediately(self, client):
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="Just a plain answer.")
        )) as mock_completion:
            res = client.post("/chat", json={"message": "hi", "session_id": None})
        assert res.json()["tool_calls"] == []
        assert mock_completion.call_count == 1

    def test_hits_max_tool_rounds_returns_limit_message(self, client):
        # An unknown tool name is unguarded AND resolved entirely inside
        # run_tool with no external calls (see test_unknown_tool_name in
        # test_tools.py) -- this test is about the harness's round-limit
        # mechanics, not any real tool's behavior or guard.
        tool_call = make_tool_call("call_x", "not_a_real_tool", {"whatever": "eggs"})
        always_calls_tool = make_message(content=None, tool_calls=[tool_call])

        with patch.object(app_module.litellm, "completion", return_value=make_completion(always_calls_tool)) as mock_completion:
            res = client.post("/chat", json={"message": "loop forever", "session_id": None})

        body = res.json()
        assert body["response"] == "Sorry, I hit my tool-call limit before finishing."
        assert mock_completion.call_count == app_module.MAX_TOOL_ROUNDS
        assert len(body["tool_calls"]) == app_module.MAX_TOOL_ROUNDS

    def test_multiple_tool_calls_in_one_round(self, client):
        # Unknown tool names so this focuses purely on "are both calls
        # recorded", with no guard or real external call involved.
        call1 = make_tool_call("c1", "not_a_real_tool_a", {"allergenic_ingredient": "butter"})
        call2 = make_tool_call("c2", "not_a_real_tool_b", {"allergenic_ingredient": "eggs"})
        first_reply = make_message(content=None, tool_calls=[call1, call2])
        second_reply = make_message(content="Here are both substitutes.", tool_calls=None)

        with patch.object(app_module.litellm, "completion", side_effect=[
            make_completion(first_reply), make_completion(second_reply),
        ]):
            res = client.post("/chat", json={"message": "butter and eggs subs?", "session_id": None})

        assert len(res.json()["tool_calls"]) == 2

    def test_unknown_tool_name_from_model_does_not_crash(self, client):
        """Models occasionally invent tool names -- the harness must survive it."""
        bad_call = make_tool_call("c1", "nonexistent_tool", {})
        first_reply = make_message(content=None, tool_calls=[bad_call])
        second_reply = make_message(content="Let me try something else.", tool_calls=None)

        with patch.object(app_module.litellm, "completion", side_effect=[
            make_completion(first_reply), make_completion(second_reply),
        ]):
            res = client.post("/chat", json={"message": "hi", "session_id": None})

        assert res.status_code == 200
        result = json.loads(res.json()["tool_calls"][0]["result"])
        assert "Unknown tool" in result["error"]


# ============================================================================
# _guarded_run_tool: narrow-classifier check for a guessed target_language
# ============================================================================
#
# Regression tests for a bug that survived several rounds of system-prompt-
# only fixes: the model kept defaulting generate_allergen_disclaimer's
# target_language to a guessed value (almost always "Spanish") when the user
# never specified a destination, by its own admission once ("the tool
# required a target language, so it defaulted to Spanish").
#
# Two string-matching versions were tried and both broke on real cases:
# checking the whole conversation caught "Spanish" but also rejected nothing
# (false negative on reuse: "Koo Thai" / "thai red curry" wrongly grounded
# "Thai" as a destination); checking only the latest message fixed that but
# then rejected "I'm traveling to France" -> target_language="French" (true
# country-to-language inference, just no shared text) -- which pushed the
# model to abandon the tool after two rejections and write an ungrounded
# card from memory, worse than the bug being fixed. The actual fix is a
# narrow LLM classifier (_language_grounded_in_message) whose only job is
# "does this message indicate this language/region" -- real understanding,
# not a lookup table, and a small enough task to be reliable even though the
# main orchestrator (juggling 3 tools and a dozen rules) isn't.

def make_classifier_response(answer: str):
    msg = MagicMock()
    msg.content = answer
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


class TestGuardedTargetLanguage:
    def test_unmentioned_language_is_rejected_without_calling_the_real_tool(self):
        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "I am allergic to shellfish and want a travel card"},
        ]
        with patch.object(app_module.litellm, "completion", return_value=make_classifier_response("NO")), \
             patch.object(app_module, "run_tool") as mock_run_tool:
            result = json.loads(app_module._guarded_run_tool(
                "generate_allergen_disclaimer", {"allergies": ["shellfish"], "target_language": "Spanish"}, messages,
            ))
        mock_run_tool.assert_not_called()
        assert "error" in result
        assert "Spanish" in result["error"]
        assert "ask" in result["suggestion"].lower()

    def test_mentioned_language_is_allowed_through(self):
        messages = [{"role": "user", "content": "I am traveling to Thailand, allergic to shellfish"}]
        with patch.object(app_module.litellm, "completion", return_value=make_classifier_response("YES")), \
             patch.object(app_module, "run_tool", return_value='{"ok": true}') as mock_run_tool:
            result = app_module._guarded_run_tool(
                "generate_allergen_disclaimer", {"allergies": ["shellfish"], "target_language": "Thai"}, messages,
            )
        mock_run_tool.assert_called_once()
        assert result == '{"ok": true}'

    def test_country_name_grounds_a_differently_spelled_language(self):
        """The actual France/French bug: the words share no text, but the
        classifier should still recognize the country implies the language."""
        messages = [{"role": "user", "content": "im travelling to france"}]
        with patch.object(app_module.litellm, "completion", return_value=make_classifier_response("YES")), \
             patch.object(app_module, "run_tool", return_value='{"ok": true}') as mock_run_tool:
            result = app_module._guarded_run_tool(
                "generate_allergen_disclaimer", {"allergies": ["coconut"], "target_language": "French"}, messages,
            )
        mock_run_tool.assert_called_once()
        assert result == '{"ok": true}'

    def test_other_tools_are_not_checked(self):
        # search_restaurant_menu is the one tool with no guard -- confirms
        # the classifier is only invoked for the two tools known to have
        # fabricated arguments live (target_language, allergenic_ingredient).
        messages = [{"role": "user", "content": "check a dish"}]
        with patch.object(app_module.litellm, "completion") as mock_completion, \
             patch.object(app_module, "run_tool", return_value='{"ok": true}') as mock_run_tool:
            app_module._guarded_run_tool(
                "search_restaurant_menu", {"restaurant_name": "X", "dish_name": "Y", "allergen": "Z"}, messages,
            )
        mock_completion.assert_not_called()
        mock_run_tool.assert_called_once()

    def test_assistant_mentioning_a_language_does_not_count_as_user_saying_it(self):
        """Only the user's own words should ground the language -- not the
        assistant's prior (possibly also-wrong) output. _latest_user_message
        only looks at role == "user", so the classifier never even sees the
        assistant's text here."""
        messages = [
            {"role": "user", "content": "I am allergic to shellfish"},
            {"role": "assistant", "content": "Here is your card in Spanish."},
        ]
        with patch.object(app_module.litellm, "completion", return_value=make_classifier_response("NO")), \
             patch.object(app_module, "run_tool") as mock_run_tool:
            app_module._guarded_run_tool(
                "generate_allergen_disclaimer", {"allergies": ["shellfish"], "target_language": "Spanish"}, messages,
            )
        mock_run_tool.assert_not_called()

    def test_restaurant_or_recipe_name_does_not_count_as_a_destination(self):
        """The Koo Thai bug: the user said 'Koo Thai' (a restaurant) and
        'thai red curry' (a recipe) several turns earlier, never a travel
        destination -- then asked for a travel card with no destination in
        that ask. The classifier only ever sees the LATEST user message, so
        it has no way to wrongly ground this even if asked."""
        messages = [
            {"role": "user", "content": "I am allergic to coconut and want to eat at Koo Thai"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "I want to make thai red curry at home"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "I'm traveling soon and want an allergy card for my allergies."},
        ]
        with patch.object(app_module.litellm, "completion") as mock_completion, \
             patch.object(app_module, "run_tool") as mock_run_tool:
            mock_completion.return_value = make_classifier_response("NO")
            result = json.loads(app_module._guarded_run_tool(
                "generate_allergen_disclaimer", {"allergies": ["coconut"], "target_language": "Thai"}, messages,
            ))
        # Confirms only the latest message reached the classifier prompt.
        sent_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
        assert "Koo Thai" not in sent_prompt
        assert "thai red curry" not in sent_prompt
        mock_run_tool.assert_not_called()
        assert "error" in result

    def test_classifier_failure_fails_closed(self):
        """If the classifier call itself errors (network, quota, etc.), the
        guard must not silently let a possibly-guessed language through."""
        messages = [{"role": "user", "content": "I am traveling to Thailand"}]
        with patch.object(app_module.litellm, "completion", side_effect=RuntimeError("down")), \
             patch.object(app_module, "run_tool") as mock_run_tool:
            result = json.loads(app_module._guarded_run_tool(
                "generate_allergen_disclaimer", {"allergies": ["shellfish"], "target_language": "Thai"}, messages,
            ))
        mock_run_tool.assert_not_called()
        assert "error" in result

    def test_end_to_end_via_chat_never_shows_an_unfounded_card(self, client):
        """Full /chat flow: the model guesses Spanish with no basis, the
        classifier says NO, the guard intercepts before the real (mocked)
        tool runs, and the harness loops to let the model ask instead."""
        bad_call = make_tool_call("c1", "generate_allergen_disclaimer", {
            "allergies": ["shellfish"], "target_language": "Spanish",
        })
        first_reply = make_message(content=None, tool_calls=[bad_call])
        second_reply = make_message(content="Where are you traveling to?", tool_calls=None)

        with patch.object(app_module.litellm, "completion", side_effect=[
            make_completion(first_reply), make_classifier_response("NO"), make_completion(second_reply),
        ]), patch.object(app_module, "run_tool") as mock_run_tool:
            res = client.post("/chat", json={
                "message": "I am allergic to shellfish and want a travel card",
                "session_id": None,
            })

        mock_run_tool.assert_not_called()
        body = res.json()
        assert body["response"] == "Where are you traveling to?"
        rejected = json.loads(body["tool_calls"][0]["result"])
        assert "error" in rejected


# ============================================================================
# _guarded_run_tool: narrow-classifier check for a fabricated allergen
# ============================================================================
#
# Regression tests for a live bug: the user said only "I want to find a
# substitute for an ingredient I'm allergic to" then "i want to make thai
# red curry" (naming a DISH, not an allergen) -- the model called
# find_ingredient_substitute with allergenic_ingredient="peanut butter",
# fabricated from nothing. The user had to point out "i havent said my
# allergy" before the model backtracked. Unlike target_language, this check
# scans the WHOLE conversation (allergies are meant to be remembered across
# turns per the system prompt), not just the latest message.

class TestGuardedAllergenicIngredient:
    def test_fabricated_ingredient_with_zero_allergy_mentioned_is_rejected(self):
        """The exact live bug: a dish was named, no allergy ever stated."""
        messages = [
            {"role": "user", "content": "I want to find a substitute for an ingredient I'm allergic to in a recipe."},
            {"role": "assistant", "content": "What ingredient are you trying to replace, and what are you making?"},
            {"role": "user", "content": "i want to make thai red curry"},
        ]
        with patch.object(app_module.litellm, "completion", return_value=make_classifier_response("NO")), \
             patch.object(app_module, "run_tool") as mock_run_tool:
            result = json.loads(app_module._guarded_run_tool(
                "find_ingredient_substitute", {"allergenic_ingredient": "peanut butter"}, messages,
            ))
        mock_run_tool.assert_not_called()
        assert "error" in result
        assert "peanut butter" in result["error"]
        assert "ask" in result["suggestion"].lower()

    def test_ingredient_stated_as_allergy_is_allowed_through(self):
        messages = [{"role": "user", "content": "I'm allergic to peanut butter, need a substitute for cookies"}]
        with patch.object(app_module.litellm, "completion", return_value=make_classifier_response("YES")), \
             patch.object(app_module, "run_tool", return_value='{"ok": true}') as mock_run_tool:
            result = app_module._guarded_run_tool(
                "find_ingredient_substitute", {"allergenic_ingredient": "peanut butter"}, messages,
            )
        mock_run_tool.assert_called_once()
        assert result == '{"ok": true}'

    def test_allergy_stated_several_turns_earlier_is_still_grounded(self):
        """Unlike target_language, this checks the WHOLE conversation --
        an allergy is a standing fact, unlike a transient destination."""
        messages = [
            {"role": "user", "content": "I'm allergic to peanut butter"},
            {"role": "assistant", "content": "Got it, noted."},
            {"role": "user", "content": "I want to make thai red curry"},
        ]
        with patch.object(app_module.litellm, "completion", return_value=make_classifier_response("YES")) as mock_completion, \
             patch.object(app_module, "run_tool", return_value='{"ok": true}') as mock_run_tool:
            app_module._guarded_run_tool(
                "find_ingredient_substitute", {"allergenic_ingredient": "peanut butter"}, messages,
            )
        mock_run_tool.assert_called_once()
        # Confirms the classifier prompt included the full conversation, not
        # just the latest message (which only mentions the dish).
        sent_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
        assert "peanut butter" in sent_prompt

    def test_naming_a_dish_that_commonly_contains_an_allergen_is_not_enough(self):
        """The prompt explicitly tells the classifier that naming a dish
        does not itself count as stating an allergy -- confirms the mocked
        NO path still blocks even when a dish strongly implies an allergen."""
        messages = [{"role": "user", "content": "I want to make pad thai"}]
        with patch.object(app_module.litellm, "completion", return_value=make_classifier_response("NO")), \
             patch.object(app_module, "run_tool") as mock_run_tool:
            result = json.loads(app_module._guarded_run_tool(
                "find_ingredient_substitute", {"allergenic_ingredient": "peanuts"}, messages,
            ))
        mock_run_tool.assert_not_called()
        assert "error" in result


# ============================================================================
# /chat: error handling
# ============================================================================

class TestChatErrorHandling:
    def test_model_exception_returns_200_with_graceful_message_not_500(self, client):
        with patch.object(app_module.litellm, "completion", side_effect=RuntimeError("auth failed")):
            res = client.post("/chat", json={"message": "hi", "session_id": None})

        assert res.status_code == 200
        body = res.json()
        assert "Model call failed" in body["response"]
        assert "RuntimeError" in body["response"]
        assert body["tool_calls"] == []
        assert body["session_id"]  # session was still created despite the failure

    def test_missing_message_field_is_422(self, client):
        res = client.post("/chat", json={"session_id": None})
        assert res.status_code == 422

    def test_empty_message_string_is_accepted(self, client):
        """Pydantic allows an empty string; the harness doesn't special-case it."""
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="I didn't get that -- can you rephrase?")
        )):
            res = client.post("/chat", json={"message": "", "session_id": None})
        assert res.status_code == 200


# ============================================================================
# /clear
# ============================================================================

class TestClear:
    def test_clear_removes_session(self, client):
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="ok")
        )):
            r = client.post("/chat", json={"message": "hi", "session_id": None})
        sid = r.json()["session_id"]
        assert sid in app_module.sessions

        res = client.post("/clear", params={"session_id": sid})
        assert res.status_code == 200
        assert sid not in app_module.sessions

    def test_clear_nonexistent_session_does_not_error(self, client):
        res = client.post("/clear", params={"session_id": "does-not-exist"})
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}

    def test_cleared_session_starts_fresh_on_next_message(self, client):
        with patch.object(app_module.litellm, "completion", return_value=make_completion(
            make_message(content="ok")
        )):
            r = client.post("/chat", json={"message": "I am allergic to peanuts", "session_id": None})
            sid = r.json()["session_id"]
            client.post("/clear", params={"session_id": sid})
            client.post("/chat", json={"message": "what are my allergies?", "session_id": sid})

        user_messages = [m["content"] for m in app_module.sessions[sid] if m["role"] == "user"]
        assert "I am allergic to peanuts" not in user_messages
        assert "what are my allergies?" in user_messages
