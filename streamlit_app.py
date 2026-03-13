from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import streamlit as st

from src.echolang.pipeline import EchoLangPipeline


DEFAULT_MODEL_PATH = "echolang_bundle.pkl"


@st.cache_resource
def load_pipeline(model_path: str) -> EchoLangPipeline:
    return EchoLangPipeline.load(model_path)


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(255, 180, 80, 0.18), transparent 36%),
                radial-gradient(circle at 90% 20%, rgba(30, 170, 170, 0.22), transparent 36%),
                linear-gradient(145deg, #f6f8fb 0%, #eaf1f7 100%);
        }

        .card {
            border: 1px solid rgba(34, 47, 62, 0.12);
            background: rgba(255, 255, 255, 0.82);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 10px 30px rgba(24, 32, 45, 0.08);
        }

        .title {
            font-size: 2rem;
            font-weight: 800;
            color: #17212b;
            letter-spacing: -0.01em;
        }

        .subtitle {
            color: #415160;
            font-size: 1rem;
            margin-top: -8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_top3(top_intents: List[Dict[str, float]]) -> None:
    for row in top_intents:
        intent = row["intent"]
        pct = row["percentage"]
        st.write(f"**{intent}** - {pct:.2f}%")
        st.progress(min(max(pct / 100.0, 0.0), 1.0))


def _render_actions(actionable: Dict[str, List[str]]) -> None:
    for intent, steps in actionable.items():
        with st.expander(f"Action plan: {intent}"):
            for idx, step in enumerate(steps, start=1):
                st.write(f"{idx}. {step}")


def main() -> None:
    st.set_page_config(page_title="EchoLang", page_icon="EL", layout="wide")
    _inject_theme()

    st.markdown('<div class="title">EchoLang - Multilingual Speech Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Speech/Text in English, Hindi, Hinglish, Tamil, and Tamil+English -> English -> 3-intent actionable routing</div>',
        unsafe_allow_html=True,
    )

    model_path = st.text_input("Model artifact path (.pkl)", value=DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        st.warning("Model file not found. Build in Kaggle and place `echolang_bundle.pkl` here.")
        st.stop()

    pipeline = load_pipeline(model_path)
    card = pipeline.model_card()

    with st.container(border=True):
        left, right = st.columns([2, 1])
        with left:
            st.write(f"**Backend:** {card['bundle']['backend']}")
            st.write(f"**Created:** {card['bundle']['created_at']}")
            st.write(f"**Version:** {card['bundle']['version']}")
        with right:
            st.write("**Training metrics**")
            metrics = card["bundle"]["training_metrics"]
            st.json(metrics)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("### Text Input")
        user_text = st.text_area(
            "Paste text request",
            height=180,
            placeholder="Example: Enakku plumber venum urgently near bus stand.",
        )
        if st.button("Analyze Text", type="primary", use_container_width=True):
            if not user_text.strip():
                st.error("Please enter text before running analysis.")
            else:
                result = pipeline.analyze_text(user_text)
                st.session_state["result"] = result

    with right_col:
        st.markdown("### Speech Input")
        audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"])
        mic_audio = st.audio_input("Record audio (optional)")
        if st.button("Analyze Speech", use_container_width=True):
            payload = None
            suffix = ".wav"

            if mic_audio is not None:
                payload = mic_audio.read()
                suffix = ".wav"
            elif audio_file is not None:
                payload = audio_file.read()
                suffix = Path(audio_file.name).suffix or ".wav"

            if not payload:
                st.error("Upload or record audio before analysis.")
            else:
                result = pipeline.analyze_audio(payload, suffix=suffix)
                st.session_state["result"] = result

    result = st.session_state.get("result")
    if result:
        st.markdown("### Analysis Output")
        with st.container(border=True):
            a, b, c = st.columns([2, 1, 1])
            a.metric("Predicted Intent", result["intent"])
            b.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
            c.metric("Urgency", result["urgency"])

            st.write("**Original text**")
            st.code(result["original_text"])
            st.write("**English text for analyzer**")
            st.code(result["translated_text"])

            st.write("**Top 3 intent probabilities**")
            _render_top3(result["top_intents"])

            st.write("**Actionable next steps**")
            _render_actions(result["actionable"])

            with st.expander("Raw JSON"):
                st.json(result)


if __name__ == "__main__":
    main()
