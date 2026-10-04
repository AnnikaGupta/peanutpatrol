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

        with patch.object(app_module.litellm, "completion", side_effect=[
            make_completion(first_reply), make_completion(second_reply),
        ]):
            res = client.post("/chat", json={"message": "substitute for butter?", "session_id": None})

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
        tool_call = make_tool_call("call_x", "find_ingredient_substitute", {"allergenic_ingredient": "eggs"})
        always_calls_tool = make_message(content=None, tool_calls=[tool_call])

        with patch.object(app_module.litellm, "completion", return_value=make_completion(always_calls_tool)) as mock_completion:
            res = client.post("/chat", json={"message": "loop forever", "session_id": None})

        body = res.json()
        assert body["response"] == "Sorry, I hit my tool-call limit before finishing."
        assert mock_completion.call_count == app_module.MAX_TOOL_ROUNDS
        assert len(body["tool_calls"]) == app_module.MAX_TOOL_ROUNDS

    def test_multiple_tool_calls_in_one_round(self, client):
        call1 = make_tool_call("c1", "find_ingredient_substitute", {"allergenic_ingredient": "butter"})
        call2 = make_tool_call("c2", "find_ingredient_substitute", {"allergenic_ingredient": "eggs"})
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
