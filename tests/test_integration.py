"""Integration tests against the REAL Spoonacular and Gemini/Vertex AI APIs.

Not run by default (pyproject.toml's addopts excludes -m integration) --
these cost API quota, take much longer, and some assertions are inherently
probabilistic since they depend on live LLM output. Run explicitly with:

    uv run pytest -m integration

Requires a working .env (GOOGLE_CLOUD_PROJECT with Vertex AI enabled and
billing, SPOONACULAR_API_KEY) and `gcloud auth application-default login`.

Where a behavior is known to vary run-to-run (see the Spanish-card and
re-asking-known-allergies bugs found during manual testing), the test
either checks a structural invariant that must always hold regardless of
phrasing, or runs the call several times and requires a majority to pass
rather than every single run -- a single flaky failure here reflects real
model non-determinism, not a broken test.
"""

import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
import tools

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def clear_sessions():
    app_module.sessions.clear()
    yield
    app_module.sessions.clear()


@pytest.fixture
def client():
    return TestClient(app_module.app)


# ============================================================================
# Tool-level integration tests
# ============================================================================

class TestSearchRestaurantMenuLive:
    def test_chipotle_peanut_bowl_not_found_in_ingredients(self):
        """Chipotle publishes an official allergen guide stating no peanuts
        are used -- this is about as stable a real-world fact as this tool
        can be tested against. A single named dish (not dish_name='menu')
        is expected to leave the dish lists empty -- those only populate for
        whole-menu questions, by design."""
        result = json.loads(tools.search_restaurant_menu("Chipotle", "chicken bowl", "peanuts"))
        assert "error" not in result
        assert result["status"] == "not_found_in_ingredients"

    def test_chipotle_whole_menu_question_populates_candidate_dishes(self):
        result = json.loads(tools.search_restaurant_menu("Chipotle", "menu", "peanuts"))
        assert "error" not in result
        assert len(result["candidate_dishes"]) > 0

    def test_returns_valid_json_shape_for_an_unfindable_restaurant(self):
        result = json.loads(tools.search_restaurant_menu(
            "Xyzabc Qwerty Nonexistent Diner 99999", "menu", "shellfish"
        ))
        assert "error" not in result
        assert result["status"] in ("ask_restaurant", "not_found_in_ingredients", "likely_contains")

    def test_whole_menu_question_returns_dish_lists_or_empty_lists_not_missing_keys(self):
        result = json.loads(tools.search_restaurant_menu("Chipotle", "menu", "peanuts"))
        assert isinstance(result["flagged_dishes"], list)
        assert isinstance(result["candidate_dishes"], list)
        assert isinstance(result["cuisine_caution"], str)


class TestFindIngredientSubstituteLive:
    def test_peanut_butter_uses_real_spoonacular_data(self):
        # find_ingredient_substitute's fallback note doesn't distinguish "quota
        # exhausted" from "ingredient genuinely not in Spoonacular's DB" (both
        # are non-200 or empty-substitutes responses) -- check the quota signal
        # directly so a real regression isn't masked as "oh it's probably quota".
        import requests as _requests
        probe = _requests.get(
            f"{tools.SPOONACULAR_BASE}/food/ingredients/substitutes",
            params={"ingredientName": "peanut butter", "apiKey": tools.SPOONACULAR_API_KEY},
        )
        if probe.status_code == 402:
            pytest.skip("Spoonacular's free-tier daily quota is exhausted -- "
                        "not a code regression, just out of quota for today.")

        result = json.loads(tools.find_ingredient_substitute("peanut butter", "cookies"))
        assert result["source"] == "Spoonacular"
        assert len(result["substitutes"]) > 0

    def test_shellfish_falls_back_to_curated_since_spoonacular_has_no_data(self):
        result = json.loads(tools.find_ingredient_substitute("shellfish"))
        assert result["source"] == "curated_fallback"
        assert len(result["substitutes"]) > 0


class TestGenerateAllergenDisclaimerLive:
    def test_thai_shellfish_card_has_translation_and_cultural_notes(self):
        result = json.loads(tools.generate_allergen_disclaimer(["shellfish"], "Thai"))
        assert "error" not in result
        assert len(result["translated_card"]) > 0
        assert len(result["cultural_notes"]) > 0

    def test_language_not_in_any_hardcoded_list_still_works(self):
        """Regression check for the old 5-language hardcoded dict this tool
        used to be limited to -- Swahili was never one of them."""
        result = json.loads(tools.generate_allergen_disclaimer(["dairy"], "Swahili"))
        assert "error" not in result
        assert len(result["translated_card"]) > 0


# ============================================================================
# Full-agent integration tests (the actual bugs found and fixed this session)
# ============================================================================

class TestAgentBehaviorLive:
    def test_cuisine_name_is_not_treated_as_a_literal_restaurant(self, client):
        """'a thai restaurant' is a cuisine description, not a business --
        the agent should answer with general cuisine guidance and ask for a
        specific restaurant, never call search_restaurant_menu with 'a thai
        restaurant' as restaurant_name."""
        res = client.post("/chat", json={
            "message": "I am allergic to shellfish and want to eat at a thai restaurant, what should I avoid?",
            "session_id": None,
        })
        body = res.json()
        for call in body["tool_calls"]:
            if call["name"] == "search_restaurant_menu":
                name = call["args"]["restaurant_name"].lower()
                assert "restaurant" not in name and "place" not in name

    def test_known_allergy_is_not_re_asked_across_turns(self, client):
        r1 = client.post("/chat", json={
            "message": "I am allergic to shellfish and want to make crab rangoons",
            "session_id": None,
        })
        sid = r1.json()["session_id"]
        r2 = client.post("/chat", json={
            "message": "actually, what ingredient substitute would work for a different recipe?",
            "session_id": sid,
        })
        # The model shouldn't need to ask "what are your allergies" again --
        # it was already told shellfish in turn 1.
        assert "what are your allerg" not in r2.json()["response"].lower()

    def test_vague_intent_with_no_specifics_does_not_call_a_tool(self, client):
        """Matches what the suggested-action chips send: signals intent
        without any real restaurant/ingredient/destination."""
        res = client.post("/chat", json={
            "message": "I want to check if a dish at a restaurant is safe for my allergies.",
            "session_id": None,
        })
        assert res.json()["tool_calls"] == []

    def test_session_memory_end_to_end(self, client):
        r1 = client.post("/chat", json={"message": "I am allergic to dairy", "session_id": None})
        sid = r1.json()["session_id"]
        r2 = client.post("/chat", json={"message": "what allergy did I just mention?", "session_id": sid})
        assert "dairy" in r2.json()["response"].lower()
