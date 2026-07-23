import os

import requests
import streamlit as st

# Base URL of the FastAPI backend, e.g. "http://127.0.0.1:8000" locally or
# "http://api:8000" when running under docker-compose. Configure via the
# API_URL environment variable (see .env.example); defaults to localhost for
# convenience when running both services on the same machine.
API_BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE_URL.rstrip('/')}/predict"


def call_api(text: str):
    if not text.strip():
        return None, None

    try:
        resp = requests.post(PREDICT_URL, json={"text": text}, timeout=30)
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach backend API at {API_BASE_URL}. Is the FastAPI service running?")
        return None, None
    except requests.exceptions.Timeout:
        st.error("Request to the backend API timed out.")
        return None, None

    if resp.status_code != 200:
        st.error(f"API error: {resp.status_code} - {resp.text}")
        return None, None

    data = resp.json()
    return data.get("label"), data.get("confidence")


st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write(
    "This app uses a fine-tuned **DistilBERT** model served via a **FastAPI backend** "
    "to classify movie reviews as positive or negative."
)

examples = [
    "I absolutely loved this movie. The acting was brilliant and the story was inspiring.",
    "This was a complete waste of time. The plot made no sense and the acting was terrible.",
    "It was okay, a bit slow in the middle but the ending was satisfying.",
]

with st.sidebar:
    st.header("Examples")
    example = st.selectbox("Choose a sample review:", ["(none)"] + examples)
    if example != "(none)":
        st.session_state["review_text"] = example

default_text = st.session_state.get("review_text", "")

review_text = st.text_area(
    "Enter a movie review:",
    value=default_text,
    height=180,
    placeholder="Type or paste a movie review here...",
)

if st.button("Analyze Sentiment"):
    label, confidence = call_api(review_text)

    if label is None:
        if review_text.strip():
            st.warning("Could not get a prediction. Check logs or API status.")
    else:
        confidence_pct = confidence * 100
        if label == "Positive":
            st.success(f"✅ Sentiment: **{label}** ({confidence_pct:.2f}% confidence)")
        else:
            st.error(f"❌ Sentiment: **{label}** ({confidence_pct:.2f}% confidence)")

        st.caption("Model: DistilBERT fine-tuned on the IMDB movie reviews dataset.")
