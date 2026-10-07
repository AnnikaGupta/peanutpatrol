# Peanut Patrol

Allergy-aware food assistant

**Live app:** https://peanutpatrol-git-387773877374.europe-west1.run.app

Welcome to Peanut Patrol, your allergy-aware food assistant.

Ordering out? You can cross-check a restaurant's menu for allergen-safe dishes, and dishes to
strictly avoid!

Cooking/Baking? Swap an unsafe ingredient for a safe one using the Spoonacular API, and find
substitutes for popular recipes.

Travelling? Communicating a moderate to severe life-threatening allergy anywhere with a
language barrier is always a challenge — specify your travel destination and generate an
allergy card to bring with you to your destination.

## Sample Queries

1. I'm allergic to shellfish, and I want to order drunken noodles from Koo Thai in the Upper
   West Side. *Expected: confirms shellfish isn't a listed ingredient, but flags that fish
   sauce or oyster sauce commonly hide in Thai stir-fries even when the dish name doesn't say
   so.*
2. I'm allergic to shellfish and I want to make crab rangoons. *Expected: a couple of real
   substitute options (e.g. hearts of palm) and a question asking which you'd like -- name one
   and you'll get back a full recipe with quantities and steps.*
3. I'm traveling to China and I'm allergic to peanuts. *Expected: a Mandarin-translated allergy
   card, cuisine-specific notes on where peanuts commonly show up, and a downloadable PNG
   version of the card.*
