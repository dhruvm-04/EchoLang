"""EchoLang package.

Provides a lightweight multilingual service-intent pipeline that can be
trained in Kaggle and exported as a single PKL artifact for Streamlit.
"""

from .pipeline import EchoLangBundle, EchoLangPipeline, build_default_dataset

__all__ = [
    "EchoLangBundle",
    "EchoLangPipeline",
    "build_default_dataset",
]
