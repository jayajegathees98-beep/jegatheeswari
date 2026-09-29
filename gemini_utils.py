"""
gemini_utils.py
----------------
All Gemini AI orchestration for PocketSmart AI: prompt construction,
calling the model, parsing its JSON output, and attaching shopping
links for Home, Party, and Jewelry recommendations.
"""
import json
import os
import re
import urllib.parse
from io import BytesIO
from typing import Optional
from urllib.parse import quote_plus

import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not API_KEY:
    raise ValueError(
        "No Google API key found. Set GOOGLE_API_KEY in your .env file. "
        "Get one free at https://aistudio.google.com"
    )

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------
class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = None


class PartyBudgetInput(BaseModel):
    total_budget: float
    num_guests: int
    party_type: str
    venue_type: Optional[str] = None
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = True
    additional_requirements: Optional[str] = None


class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def extract_json_from_response(text: str) -> dict:
    """Gemini sometimes wraps JSON in ```json fences or adds preamble text."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse AI response as JSON: {e}\nRaw response:\n{text[:1000]}")


def _link(base: str, search_terms: str) -> str:
    return f"{base}{quote_plus(search_terms)}"


CATEGORY_PLATFORMS = {
    "lighting": ["amazon", "flipkart", "ikea"],
    "fans": ["amazon", "flipkart", "ikea"],
    "furniture": ["amazon", "flipkart", "ikea"],
    "dining": ["amazon", "flipkart", "ikea"],
    "venue": ["google", "booking", "makemytrip", "oyorooms", "nobroker"],
    "catering": ["swiggy", "zomato", "bigbasket"],
    "food": ["swiggy", "zomato", "bigbasket"],
    "decoration": ["amazon", "flipkart", "meesho", "myntra"],
    "entertainment": ["bookmyshow", "amazon", "flipkart"],
}
DEFAULT_PLATFORMS = ["amazon", "flipkart"]

PLATFORM_URLS = {
    "amazon": "https://www.amazon.in/s?k=",
    "flipkart": "https://www.flipkart.com/search?q=",
    "ikea": "https://www.ikea.com/in/en/search/?q=",
    "myntra": "https://www.myntra.com/search/",  # + term appended directly
    "meesho": "https://www.meesho.com/search?q=",
    "swiggy": "https://www.swiggy.com/search?query=",
    "zomato": "https://www.zomato.com/search?q=",
    "bigbasket": "https://www.bigbasket.com/ps/?q=",
    "google": "https://www.google.com/search?q=",
    "booking": "https://www.booking.com/search.html?ss=",
    "makemytrip": "https://www.makemytrip.com/hotels/hotel-listing/?searchtext=",
    "oyorooms": "https://www.oyorooms.com/search?location=",
    "nobroker": "https://www.nobroker.in/property/search?searchParam=",
    "bookmyshow": "https://in.bookmyshow.com/explore/home/search?q=",
    "bluestone": "https://www.bluestone.com/search.html?query=",
    "tanishq": "https://www.tanishq.co.in/search?q=",
    "caratlane": "https://www.caratlane.com/search?q=",
    "melorra": "https://www.melorra.com/search?q=",
}


def attach_shopping_links(items: list, category_name: str) -> None:
    """Mutates each item dict in place, adding a shopping_links map."""
    cat_key = category_name.lower()
    matched_key = next((k for k in CATEGORY_PLATFORMS if k in cat_key), None)
    platforms = CATEGORY_PLATFORMS.get(matched_key, DEFAULT_PLATFORMS)

    for item in items:
        search_terms = item.get("search_terms", item.get("name", ""))
        if not search_terms:
            continue
        item["shopping_links"] = {
            p: _link(PLATFORM_URLS[p], search_terms) for p in platforms if p in PLATFORM_URLS
        }


def _summarize_categories(result: dict) -> None:
    """Adds a calculation_table with per-category totals & % of budget."""
    total_budget = result.get("total_budget", 0) or 0
    categories = {}
    for cat in result.get("budget_breakdown", []):
        name = cat.get("category", "Misc")
        items = cat.get("items", [])
        cat_total = sum(i.get("estimated_price", 0) * i.get("quantity", 1) for i in items)
        categories[name] = {
            "category": name,
            "items_count": len(items),
            "total_cost": round(cat_total, 2),
            "percentage_of_budget": round((cat_total / total_budget) * 100, 1) if total_budget else 0,
        }
    result["calculation_table"] = list(categories.values())


# ---------------------------------------------------------------------------
# Scenario 1: Home Interior Planner
# ---------------------------------------------------------------------------
def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    rooms = []
    if budget_input.has_living_room:
        rooms.append("Living Room")
    if budget_input.has_kitchen:
        rooms.append("Kitchen")
    if budget_input.has_bedroom:
        rooms.append("Bedroom")

    prompt = f"""
I need interior design product recommendations for a home in India with a total budget of ₹{budget_input.total_budget:.2f}.

Requirements:
- {budget_input.num_lights} lights/lighting fixtures
- {budget_input.num_fans} ceiling fans
- {budget_input.num_furniture} furniture pieces
- {budget_input.num_dining_tables} dining tables

Rooms to consider: {', '.join(rooms) if rooms else 'Not specified'}
Additional requirements: {budget_input.additional_requirements or 'None'}

Give real, realistically priced product recommendations available in India (Indian brands and INR pricing).
Include short "search_terms" for each item, suitable for searching on Indian shopping sites.

Respond with ONLY valid JSON, no preamble, no markdown fences, in exactly this structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "Lighting",
      "allocation": 0.0,
      "items": [
        {{"name": "", "description": "", "estimated_price": 0.0, "quantity": 0, "search_terms": ""}}
      ]
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": ["", ""]
}}

Ensure total costs stay within budget.
"""
    response = model.generate_content(prompt)
    result = extract_json_from_response(response.text)

    for category in result.get("budget_breakdown", []):
        attach_shopping_links(category.get("items", []), category.get("category", ""))

    _summarize_categories(result)
    return result


# ---------------------------------------------------------------------------
# Scenario 2: Party Planner
# ---------------------------------------------------------------------------
def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    prompt = f"""
I need party planning recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.

Party details:
- Type: {budget_input.party_type}
- Number of guests: {budget_input.num_guests}
- Venue type: {budget_input.venue_type or 'Not specified'}
- Catering needed: {"Yes" if budget_input.needs_catering else "No"}
- Decoration needed: {"Yes" if budget_input.needs_decoration else "No"}
- Entertainment needed: {"Yes" if budget_input.needs_entertainment else "No"}
Additional requirements: {budget_input.additional_requirements or 'None'}

Give a detailed budget breakdown with realistic INR prices, using services common in India
(catering, decoration, entertainment, and venue options). Include short "search_terms" for
each item suitable for Indian platforms like Swiggy, Zomato, BookMyShow, OYO.

Respond with ONLY valid JSON, no preamble, no markdown fences, in exactly this structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "Catering",
      "allocation": 0.0,
      "items": [
        {{"name": "", "description": "", "estimated_price": 0.0, "quantity": 0, "search_terms": ""}}
      ]
    }}
  ],
  "venue_suggestions": [
    {{"name": "", "type": "", "capacity": 0, "estimated_cost": 0.0, "search_terms": ""}}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": ["", ""]
}}

Ensure all costs are in INR and the total does not exceed the given budget.
"""
    response = model.generate_content(prompt)
    result = extract_json_from_response(response.text)

    for category in result.get("budget_breakdown", []):
        attach_shopping_links(category.get("items", []), category.get("category", ""))
    attach_shopping_links(result.get("venue_suggestions", []), "venue")

    _summarize_categories(result)
    return result


# ---------------------------------------------------------------------------
# Scenario 3: Jewelry Planner (supports optional outfit image)
# ---------------------------------------------------------------------------
def get_jewelry_recommendations(budget_input: JewelryBudgetInput, image_path: Optional[str] = None) -> dict:
    base_prompt = f"""
I need jewelry recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.

Occasion: {budget_input.occasion}
Preferences: {budget_input.preferences or 'Not specified'}

Provide only India-relevant styles, availability, and price ranges in INR.
Keep prices realistic for Indian jewelry brands. Include short "search_terms" for each item.
"""

    if image_path:
        img = Image.open(image_path)
        prompt = base_prompt + """
An image of the outfit is attached. Analyze its colors and style, and suggest jewelry that
complements it, considering color coordination and occasion appropriateness.

Respond with ONLY valid JSON, no preamble, no markdown fences, in exactly this structure:
{
  "outfit_analysis": {"colors": [], "style": "", "formality": ""},
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {"item_type": "", "description": "", "style": "", "estimated_price": 0.0, "search_terms": ""}
  ],
  "remaining_budget": 0.0,
  "styling_tips": ["", ""]
}
"""
        response = model.generate_content([prompt, img])
    else:
        prompt = base_prompt + """
Respond with ONLY valid JSON, no preamble, no markdown fences, in exactly this structure:
{
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {"item_type": "", "description": "", "style": "", "estimated_price": 0.0, "search_terms": ""}
  ],
  "remaining_budget": 0.0,
  "styling_tips": ["", ""]
}
"""
        response = model.generate_content(prompt)

    result = extract_json_from_response(response.text)

    jewelry_platforms = ["amazon", "flipkart", "bluestone", "tanishq", "caratlane", "melorra", "meesho"]
    for item in result.get("jewelry_recommendations", []):
        search_terms = item.get("search_terms", item.get("item_type", ""))
        if search_terms:
            item["shopping_links"] = {
                p: _link(PLATFORM_URLS[p], search_terms) for p in jewelry_platforms
            }

    return result
