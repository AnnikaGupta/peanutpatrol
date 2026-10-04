"""Unit tests for tools.py. All external calls (litellm, requests) are mocked --
no network access, no API cost, deterministic. See test_integration.py for tests
against the real APIs.
"""

import json
from unittest.mock import MagicMock, patch

import litellm
import pytest
import requests

import tools


@pytest.fixture(autouse=True)
def gcp_project_env(monkeypatch):
    """search_restaurant_menu and generate_allergen_disclaimer check
    os.getenv('GOOGLE_CLOUD_PROJECT') before doing any work. These tests mock
    the grounded call itself, so the value just needs to be truthy -- scoped
    to this file only so it doesn't clobber the real project in
    test_integration.py."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")


# ============================================================================
# Helpers
# ============================================================================

def make_grounded_response(content: str):
    """Build a fake litellm.completion() return value whose
    .choices[0].message.content is the given string."""
    message = MagicMock()
    message.content = content
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    return resp


# ============================================================================
# _grounded_json_call
# ============================================================================

class TestGroundedJsonCall:
    def test_parses_clean_json(self):
        with patch.object(tools.litellm, "completion", return_value=make_grounded_response(
            '{"status": "ok"}'
        )):
            result = tools._grounded_json_call("prompt")
        assert result == {"status": "ok"}

    def test_strips_markdown_fences(self):
        with patch.object(tools.litellm, "completion", return_value=make_grounded_response(
            '```json\n{"status": "ok"}\n```'
        )):
            result = tools._grounded_json_call("prompt")
        assert result == {"status": "ok"}

    def test_strips_bare_fences_without_json_tag(self):
        with patch.object(tools.litellm, "completion", return_value=make_grounded_response(
            '```\n{"status": "ok"}\n```'
        )):
            result = tools._grounded_json_call("prompt")
        assert result == {"status": "ok"}

    def test_returns_none_on_invalid_json(self):
        with patch.object(tools.litellm, "completion", return_value=make_grounded_response(
            "Sorry, I can't help with that."
        )):
            result = tools._grounded_json_call("prompt")
        assert result is None

    def test_returns_none_on_empty_content(self):
        with patch.object(tools.litellm, "completion", return_value=make_grounded_response(None)):
            result = tools._grounded_json_call("prompt")
        assert result is None

    def test_retries_once_on_timeout_then_succeeds(self):
        good = make_grounded_response('{"status": "ok"}')
        with patch.object(
            tools.litellm, "completion",
            side_effect=[litellm.Timeout("timed out", "model", "provider"), good],
        ) as mock_completion:
            result = tools._grounded_json_call("prompt")
        assert result == {"status": "ok"}
        assert mock_completion.call_count == 2

    def test_raises_after_exhausting_retries(self):
        with patch.object(
            tools.litellm, "completion",
            side_effect=litellm.Timeout("timed out", "model", "provider"),
        ) as mock_completion:
            with pytest.raises(litellm.Timeout):
                tools._grounded_json_call("prompt")
        assert mock_completion.call_count == 2  # initial attempt + 1 retry

    def test_non_timeout_exception_propagates_immediately_no_retry(self):
        with patch.object(
            tools.litellm, "completion", side_effect=ValueError("boom")
        ) as mock_completion:
            with pytest.raises(ValueError):
                tools._grounded_json_call("prompt")
        assert mock_completion.call_count == 1


# ============================================================================
# search_restaurant_menu
# ============================================================================

class TestSearchRestaurantMenu:
    def test_missing_gcp_project_returns_error(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        result = json.loads(tools.search_restaurant_menu("Chipotle", "chicken bowl", "peanuts"))
        assert "error" in result
        assert "suggestion" in result

    def test_likely_contains_full_shape(self):
        parsed = {
            "status": "likely_contains",
            "headline": "Contains peanuts in several dishes.",
            "explanation": "...",
            "flagged_dishes": ["Pad Thai"],
            "candidate_dishes": ["Green Curry"],
            "cuisine_caution": "Peanut oil is common in stir-fries.",
            "location_ambiguous": False,
            "location_options": [],
        }
        with patch.object(tools, "_grounded_json_call", return_value=parsed):
            result = json.loads(tools.search_restaurant_menu("Thai Villa", "menu", "peanuts", "Chicago"))

        assert result["status"] == "likely_contains"
        assert result["restaurant"] == "Thai Villa"
        assert result["location"] == "Chicago"
        assert result["flagged_dishes"] == ["Pad Thai"]
        assert result["candidate_dishes"] == ["Green Curry"]
        assert result["cuisine_caution"] == "Peanut oil is common in stir-fries."
        assert result["source"] == "gemini_web_search"
        assert "warning" in result

    def test_not_found_in_ingredients(self):
        parsed = {"status": "not_found_in_ingredients", "headline": "h", "explanation": "e",
                   "flagged_dishes": [], "candidate_dishes": ["Chicken Bowl"],
                   "cuisine_caution": "", "location_ambiguous": False, "location_options": []}
        with patch.object(tools, "_grounded_json_call", return_value=parsed):
            result = json.loads(tools.search_restaurant_menu("Chipotle", "chicken bowl", "peanuts"))
        assert result["status"] == "not_found_in_ingredients"
        assert result["candidate_dishes"] == ["Chicken Bowl"]

    def test_location_ambiguous_surfaces_options(self):
        parsed = {
            "status": "likely_contains", "headline": "Multiple locations found.",
            "explanation": "e", "flagged_dishes": [], "candidate_dishes": [],
            "cuisine_caution": "", "location_ambiguous": True,
            "location_options": ["New York, NY", "Fishers, IN"],
        }
        with patch.object(tools, "_grounded_json_call", return_value=parsed):
            result = json.loads(tools.search_restaurant_menu("Thai Villa", "menu", "shellfish"))
        assert result["location_ambiguous"] is True
        assert result["location_options"] == ["New York, NY", "Fishers, IN"]

    def test_no_location_given_defaults_location_field_empty(self):
        parsed = {"status": "ask_restaurant", "headline": "h", "explanation": "e",
                   "flagged_dishes": [], "candidate_dishes": [], "cuisine_caution": "",
                   "location_ambiguous": False, "location_options": []}
        with patch.object(tools, "_grounded_json_call", return_value=parsed):
            result = json.loads(tools.search_restaurant_menu("Koo Thai", "pad thai", "peanuts"))
        assert result["location"] == ""

    def test_grounded_call_returns_none_falls_back_to_ask_restaurant(self):
        with patch.object(tools, "_grounded_json_call", return_value=None):
            result = json.loads(tools.search_restaurant_menu("Xyz Diner", "menu", "peanuts"))
        assert result["status"] == "ask_restaurant"
        assert "suggestion" in result

    def test_exception_during_call_returns_graceful_error(self):
        with patch.object(tools, "_grounded_json_call", side_effect=RuntimeError("network down")):
            result = json.loads(tools.search_restaurant_menu("Chipotle", "bowl", "peanuts"))
        assert "error" in result
        assert "network down" in result["error"]
        assert "suggestion" in result

    def test_missing_fields_in_model_response_default_sensibly(self):
        """If the model's JSON is missing a key (shouldn't happen given the
        prompt, but models aren't 100% reliable), nothing should KeyError."""
        with patch.object(tools, "_grounded_json_call", return_value={}):
            result = json.loads(tools.search_restaurant_menu("Chipotle", "bowl", "peanuts"))
        assert result["status"] == "ask_restaurant"
        assert result["flagged_dishes"] == []
        assert result["candidate_dishes"] == []
        assert result["cuisine_caution"] == ""


# ============================================================================
# find_ingredient_substitute
# ============================================================================

class TestFindIngredientSubstitute:
    def test_spoonacular_success_path(self):
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {
            "status": "success",
            "substitutes": ["1 cup = 1 cup tahini"],
        }
        with patch.object(tools.requests, "get", return_value=fake_resp):
            result = json.loads(tools.find_ingredient_substitute("peanut butter", "cookies"))

        assert result["source"] == "Spoonacular"
        assert result["substitutes"] == ["1 cup = 1 cup tahini"]
        assert result["recipe_type"] == "cookies"

    def test_recipe_context_defaults_to_general(self):
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {"status": "success", "substitutes": ["x"]}
        with patch.object(tools.requests, "get", return_value=fake_resp):
            result = json.loads(tools.find_ingredient_substitute("butter"))
        assert result["recipe_type"] == "general"

    def test_spoonacular_failure_status_falls_back_to_curated(self):
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {"status": "failure", "message": "not found"}
        with patch.object(tools.requests, "get", return_value=fake_resp):
            result = json.loads(tools.find_ingredient_substitute("shellfish"))
        assert result["source"] == "curated_fallback"
        assert len(result["substitutes"]) > 0

    def test_spoonacular_non_200_falls_back_to_curated(self):
        fake_resp = MagicMock(status_code=402)  # quota exceeded, Spoonacular's real behavior
        with patch.object(tools.requests, "get", return_value=fake_resp):
            result = json.loads(tools.find_ingredient_substitute("eggs"))
        assert result["source"] == "curated_fallback"

    def test_network_exception_falls_back_to_curated_with_note(self):
        with patch.object(tools.requests, "get", side_effect=requests.ConnectionError("no route")):
            result = json.loads(tools.find_ingredient_substitute("peanut butter"))
        assert result["source"] == "curated_fallback"
        assert "no route" in result["note"] or "Network error" in result["note"]

    def test_unexpected_exception_returns_error_json(self):
        with patch.object(tools.requests, "get", side_effect=ValueError("unexpected")):
            result = json.loads(tools.find_ingredient_substitute("butter"))
        assert "error" in result


class TestGetCommonSubstitutes:
    def test_exact_key_match(self):
        subs = tools.get_common_substitutes("peanut butter")
        assert any("sunflower" in s["substitute"] for s in subs)

    def test_partial_match_ingredient_contains_key(self):
        subs = tools.get_common_substitutes("creamy peanut butter")
        assert any("sunflower" in s["substitute"] for s in subs)

    def test_partial_match_key_contains_ingredient_phrase(self):
        subs = tools.get_common_substitutes("dairy milk")
        assert any("oat milk" in s["substitute"] for s in subs)

    def test_shellfish_has_curated_entry(self):
        subs = tools.get_common_substitutes("shellfish")
        assert any("mushroom" in s["substitute"] for s in subs)

    def test_unknown_ingredient_returns_generic_fallback(self):
        subs = tools.get_common_substitutes("dragonfruit extract")
        assert len(subs) == 1
        assert "No common substitute found" in subs[0]["reason"]

    def test_case_insensitive(self):
        subs = tools.get_common_substitutes("PEANUT BUTTER")
        assert any("sunflower" in s["substitute"] for s in subs)


# ============================================================================
# generate_allergen_disclaimer
# ============================================================================

class TestGenerateAllergenDisclaimer:
    def test_missing_gcp_project_returns_error(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        result = json.loads(tools.generate_allergen_disclaimer(["peanuts"], "Spanish"))
        assert "error" in result

    def test_success_shape(self):
        parsed = {
            "translated_card": "Tengo alergia a los cacahuetes.",
            "cultural_notes": ["Mole often contains peanuts."],
        }
        with patch.object(tools, "_grounded_json_call", return_value=parsed):
            result = json.loads(tools.generate_allergen_disclaimer(["peanuts"], "Spanish"))

        assert result["language"] == "Spanish"
        assert result["allergies"] == ["peanuts"]
        assert result["translated_card"] == "Tengo alergia a los cacahuetes."
        assert result["cultural_notes"] == ["Mole often contains peanuts."]
        assert result["source"] == "gemini_web_search"
        assert "warning" in result

    def test_multiple_allergies_passed_through(self):
        parsed = {"translated_card": "x", "cultural_notes": []}
        with patch.object(tools, "_grounded_json_call", return_value=parsed):
            result = json.loads(tools.generate_allergen_disclaimer(["peanuts", "dairy", "shellfish"], "Thai"))
        assert result["allergies"] == ["peanuts", "dairy", "shellfish"]

    def test_grounded_call_returns_none_gives_empty_card_with_note(self):
        with patch.object(tools, "_grounded_json_call", return_value=None):
            result = json.loads(tools.generate_allergen_disclaimer(["peanuts"], "Thai"))
        assert result["translated_card"] == ""
        assert result["cultural_notes"] == []
        assert "note" in result

    def test_exception_returns_error_with_fallback_text(self):
        with patch.object(tools, "_grounded_json_call", side_effect=RuntimeError("down")):
            result = json.loads(tools.generate_allergen_disclaimer(["peanuts", "dairy"], "Thai"))
        assert "error" in result
        assert "peanuts, dairy" in result["fallback"]

    def test_empty_allergies_list_does_not_crash(self):
        parsed = {"translated_card": "x", "cultural_notes": []}
        with patch.object(tools, "_grounded_json_call", return_value=parsed):
            result = json.loads(tools.generate_allergen_disclaimer([], "Thai"))
        assert result["allergies"] == []


# ============================================================================
# run_tool (dispatch + error handling)
# ============================================================================

class TestRunTool:
    def test_unknown_tool_name(self):
        result = json.loads(tools.run_tool("make_me_a_sandwich", {}))
        assert "Unknown tool" in result["error"]
        assert "search_restaurant_menu" in result["error"]

    def test_missing_required_argument(self):
        result = json.loads(tools.run_tool("find_ingredient_substitute", {}))
        assert "Bad arguments" in result["error"]

    def test_unexpected_keyword_argument(self):
        result = json.loads(tools.run_tool("find_ingredient_substitute", {"wrong_arg": "x"}))
        assert "Bad arguments" in result["error"]

    def test_valid_call_routes_to_correct_function(self):
        with patch.object(tools, "TOOL_MAP", {"find_ingredient_substitute": lambda **kw: json.dumps({"ok": kw})}):
            result = json.loads(tools.run_tool("find_ingredient_substitute", {"allergenic_ingredient": "eggs"}))
        assert result["ok"] == {"allergenic_ingredient": "eggs"}

    def test_general_exception_inside_tool_is_caught(self):
        def boom(**kwargs):
            raise RuntimeError("tool blew up")
        with patch.object(tools, "TOOL_MAP", {"find_ingredient_substitute": boom}):
            result = json.loads(tools.run_tool("find_ingredient_substitute", {"allergenic_ingredient": "eggs"}))
        assert "Tool execution failed" in result["error"]
        assert "tool blew up" in result["error"]

    def test_all_three_real_tools_are_registered(self):
        assert set(tools.TOOL_MAP.keys()) == {
            "search_restaurant_menu", "find_ingredient_substitute", "generate_allergen_disclaimer",
        }


# ============================================================================
# TOOLS schema sanity (what the model actually sees)
# ============================================================================

class TestToolsSchema:
    def test_every_tool_map_entry_has_a_schema(self):
        schema_names = {t["function"]["name"] for t in tools.TOOLS}
        assert schema_names == set(tools.TOOL_MAP.keys())

    def test_no_schema_description_anchors_on_spanish(self):
        """Regression test: generate_allergen_disclaimer's target_language
        description used to list 'Spanish' as its first example, which
        anchored the model into guessing Spanish by default when the user
        hadn't specified a destination -- found via live testing, not
        reproducible with a mocked unit test, but this at least guards
        against literally reintroducing the same anchor."""
        for t in tools.TOOLS:
            for prop in t["function"]["parameters"]["properties"].values():
                assert "spanish" not in prop.get("description", "").lower()

    def test_location_and_target_language_required_or_optional_as_expected(self):
        by_name = {t["function"]["name"]: t["function"] for t in tools.TOOLS}
        restaurant_params = by_name["search_restaurant_menu"]["parameters"]
        assert "location" not in restaurant_params["required"]
        disclaimer_params = by_name["generate_allergen_disclaimer"]["parameters"]
        assert "target_language" in disclaimer_params["required"]
