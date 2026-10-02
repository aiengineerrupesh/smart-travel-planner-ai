import os

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()
st.set_page_config(page_title="TripGenie AI", page_icon="🌍", layout="wide")


# ---------- Keys (works locally with .env and on Streamlit Cloud with secrets) ----------
def get_key(name: str):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None


OPENAI_KEY = get_key("OPENAI_API_KEY")
UNSPLASH_KEY = get_key("UNSPLASH_API_KEY")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY


# ---------- Structured output for places ----------
class Place(BaseModel):
    name: str
    description: str


class PlaceList(BaseModel):
    places: list[Place]


# ---------- Unsplash image ----------
@st.cache_data(show_spinner=False)
def get_place_image(query: str):
    if not UNSPLASH_KEY:
        return None
    try:
        r = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "client_id": UNSPLASH_KEY, "per_page": 1},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0]["urls"]["regular"] if results else None
    except Exception:
        return None


# ---------- LLM chains ----------
def build_chains(llm):
    parser = StrOutputParser()

    places_chain = (
        ChatPromptTemplate.from_template(
            "List the 6 top tourist places in {location}. "
            "For each give the place name and a one-line description."
        )
        | llm.with_structured_output(PlaceList)
    )
    hotels_chain = (
        ChatPromptTemplate.from_template(
            "Suggest 5 good {budget_level} hotels in {location} in bullet points, "
            "with approximate price per night."
        )
        | llm
        | parser
    )
    itinerary_chain = (
        ChatPromptTemplate.from_template(
            "Create a detailed {days}-day itinerary for {location} covering these places:\n"
            "{places}\n\nOrganise by Day 1, Day 2... with morning, afternoon and evening."
        )
        | llm
        | parser
    )
    budget_chain = (
        ChatPromptTemplate.from_template(
            "Estimate the total budget for this {budget_level} trip in INR. "
            "Break it down into stay, food, transport, activities.\n\nItinerary:\n{itinerary}"
        )
        | llm
        | parser
    )
    tips_chain = (
        ChatPromptTemplate.from_template(
            "Give 6 useful, practical travel tips for visiting {location}."
        )
        | llm
        | parser
    )
    return places_chain, hotels_chain, itinerary_chain, budget_chain, tips_chain


# ---------- UI ----------
st.title("🌍 TripGenie AI")
st.markdown("Plan your trip with AI ✈️ — places, hotels, itinerary, budget and tips in one click.")

c1, c2, c3 = st.columns([3, 1, 2])
with c1:
    location = st.text_input("📍 Enter Location", placeholder="e.g. Goa, Manali, Paris")
with c2:
    days = st.number_input("🗓️ Days", min_value=1, max_value=10, value=3)
with c3:
    budget_level = st.selectbox("💰 Budget", ["budget", "mid-range", "luxury"], index=1)

if st.button("🚀 Plan My Trip"):
    if not location.strip():
        st.warning("Please enter a location")
        st.stop()
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY not found. Add it to your .env file or Streamlit secrets.")
        st.stop()

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        places_c, hotels_c, itin_c, budget_c, tips_c = build_chains(llm)

        with st.spinner("Planning your trip..."):
            place_data = places_c.invoke({"location": location})
            places_text = "\n".join(f"- {p.name}: {p.description}" for p in place_data.places)

            hotels = hotels_c.invoke({"location": location, "budget_level": budget_level})
            itinerary = itin_c.invoke(
                {"location": location, "days": days, "places": places_text}
            )
            budget = budget_c.invoke({"itinerary": itinerary, "budget_level": budget_level})
            tips = tips_c.invoke({"location": location})
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()

    # ---------- Output ----------
    st.subheader("📍 Top Places")
    cols = st.columns(3)
    for i, place in enumerate(place_data.places[:6]):
        with cols[i % 3]:
            img = get_place_image(f"{place.name} {location}")
            if img:
                st.image(img, width="stretch")
            st.markdown(f"### {place.name}")
            st.caption(place.description)

    st.divider()
    st.subheader("🏨 Recommended Hotels")
    st.write(hotels)
    st.subheader(f"🗓️ {days}-Day Itinerary")
    st.write(itinerary)
    st.subheader("💰 Budget Estimate")
    st.write(budget)
    st.subheader("💡 Travel Tips")
    st.write(tips)
    st.success("✅ Trip planning completed!")
