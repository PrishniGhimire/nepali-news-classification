"""Streamlit demo: paste a Nepali news article and see its predicted TOPIC.

Run:  streamlit run app.py

This app classifies the topic of a text (politics, sports, economy, ...). It does not
translate articles and it does not check whether news is true.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import MISSING_MODEL_HELP, load_predictor
from src.text_utils import clean_text

st.set_page_config(page_title="Nepali news topic classifier", page_icon="📰")


def find_models() -> list[str]:
    found = [str(p) for p in sorted(Path("outputs").glob("*/best")) if (p / "config.json").exists()]
    found += [str(p) for p in sorted(Path("outputs").glob("*/model.joblib"))]
    return found


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(path: str):
    return load_predictor(path)


st.title("Nepali news topic classifier")
st.caption(
    "Paste a Nepali news article to see which topic category a trained model predicts. "
    "This tool only classifies the **topic**. It does not translate the article or check whether the news is true."
)

models = find_models()
if not models:
    st.error("No trained model found in the `outputs/` folder.")
    st.code(MISSING_MODEL_HELP.format(path="outputs/..."), language="text")
    st.stop()

model_path = st.sidebar.selectbox("Model", models, help="Models found under outputs/")
top_k = st.sidebar.slider("Categories to show", 3, 10, 5)
st.sidebar.caption("Scores are model probabilities, not guarantees. Labels come from the training dataset's sections.")

text = st.text_area("Article text (Nepali)", height=220, placeholder="यहाँ समाचार टाँस्नुहोस्...")

if st.button("Classify", type="primary"):
    cleaned = clean_text(text)
    if len(cleaned) < 10:
        st.warning("Please paste a longer piece of Nepali text.")
    else:
        try:
            predict_proba, labels = get_predictor(model_path)
        except FileNotFoundError as err:
            st.error(str(err))
            st.stop()
        probs = predict_proba([cleaned])[0]
        order = probs.argsort()[::-1][:top_k]
        st.subheader(f"Predicted topic: {labels[order[0]]}")
        st.write(f"Confidence: {probs[order[0]]:.1%}")
        table = pd.DataFrame({"topic": [labels[i] for i in order], "score": [float(probs[i]) for i in order]})
        st.bar_chart(table, x="topic", y="score")
        st.dataframe(table.assign(score=table["score"].map("{:.3f}".format)), hide_index=True)
