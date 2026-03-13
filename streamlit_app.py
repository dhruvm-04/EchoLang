from __future__ import annotations

import os

import streamlit as st

from src.echolang.pipeline import EchoLangPipeline


DEFAULT_MODEL_PATH = "echolang_bundle.pkl"


@st.cache_resource
def load_pipeline(model_path: str) -> EchoLangPipeline:
    return EchoLangPipeline.load(model_path)


def main() -> None:
    st.set_page_config(page_title="EchoLang", page_icon="EL", layout="centered")
    st.title("EchoLang Service Intent Demo")
    st.caption("Load a Kaggle-generated PKL bundle and run multilingual intent analysis.")

    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        st.warning("Model file not found. Upload or place echolang_bundle.pkl in the app folder.")
        st.stop()

    pipeline = load_pipeline(model_path)

    user_text = st.text_area("Enter request", placeholder="Example: Enakku plumber venum urgently")
    if st.button("Analyze", type="primary"):
        if not user_text.strip():
            st.error("Please enter a request.")
            st.stop()

        result = pipeline.analyze_text(user_text)
        st.subheader("Result")
        st.write(f"Intent: **{result['intent']}**")
        st.write(f"Confidence: **{result['confidence']:.2%}**")
        st.write(f"Urgency: **{result['urgency']}**")
        st.write(f"Detected Scripts: **{', '.join(result['detected_languages'])}**")
        st.json(result["top_matches"])


if __name__ == "__main__":
    main()
