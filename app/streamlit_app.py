import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
with open("../models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Title and UI
st.set_page_config(page_title="Finance Scorecard", layout="centered")
st.title("📈 Finance Scorecard")
st.markdown("Enter a financial news headline and predict the sentiment:")

# Input box
headline = st.text_input("Enter headline:", "Stocks rally after inflation data surprises")

# Predict button
if st.button("Predict Sentiment"):
    if headline.strip() == "":
        st.warning("Please enter a valid headline.")
    else:
        X = vectorizer.transform([headline])
        pred = model.predict(X)[0]

        # Map prediction to labels
        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }
        st.success(f"🧠 Predicted Sentiment: **{sentiment_map.get(pred, 'Unknown')}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Fusion Six")