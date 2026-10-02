# 🌍 TripGenie AI

AI-powered travel planner. Enter a destination, number of days and budget, and get top places (with live images), hotels, a day-wise itinerary, budget estimate and travel tips.

## Features
- Structured LLM output (Pydantic) for clean place names and images
- Chained prompts with LangChain (LCEL): places -> itinerary -> budget
- Live destination photos via Unsplash API
- Adjustable days and budget level

## Tech Stack
Python, LangChain, OpenAI GPT-4o-mini, Streamlit, Unsplash API

## Setup
```bash
git clone https://github.com/aiengineerrupesh/smart-travel-planner-ai.git
cd smart-travel-planner-ai
pip install -r requirements.txt
```
Copy `.env.example` to `.env`, add your keys, then run:
```bash
streamlit run app.py
```
