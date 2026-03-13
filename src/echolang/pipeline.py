from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


SERVICE_INTENTS: Dict[str, Dict[str, List[str] | str]] = {
    "Emergency Services": {
        "description": "Emergency support for ambulance police or fire incidents.",
        "keywords": [
            "emergency",
            "urgent",
            "ambulance",
            "fire",
            "police",
            "accident",
            "danger",
        ],
    },
    "Healthcare": {
        "description": "Doctor appointments medicine hospital and treatment requests.",
        "keywords": [
            "doctor",
            "hospital",
            "medicine",
            "checkup",
            "nurse",
            "treatment",
            "clinic",
        ],
    },
    "Home Maintenance": {
        "description": "Plumbing electrical and home repair service requests.",
        "keywords": [
            "plumber",
            "electrician",
            "repair",
            "leak",
            "pipe",
            "power",
            "maintenance",
        ],
    },
    "Transportation": {
        "description": "Taxi ride pickup drop and local transport requests.",
        "keywords": [
            "taxi",
            "cab",
            "driver",
            "pickup",
            "drop",
            "transport",
            "vehicle",
        ],
    },
    "Cleaning Services": {
        "description": "Home office deep cleaning sanitization and housekeeping.",
        "keywords": [
            "cleaning",
            "sanitize",
            "housekeeping",
            "mop",
            "dust",
            "wash",
            "deep clean",
        ],
    },
    "General Services": {
        "description": "Generic help and assistance requests that do not fit other categories.",
        "keywords": [
            "help",
            "support",
            "service",
            "assist",
            "guidance",
        ],
    },
}


TEMPLATES = [
    "I need {keyword} service now",
    "Please arrange {keyword} near me",
    "Need {keyword} in my area",
    "Can someone help with {keyword}",
    "Mujhe {keyword} chahiye",
    "Enakku {keyword} venum",
    "{keyword} urgent required",
]


def normalize_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def detect_script_languages(text: str) -> List[str]:
    patterns = {
        "hi": r"[\u0900-\u097F]+",
        "ta": r"[\u0B80-\u0BFF]+",
        "en": r"[A-Za-z]+",
    }
    detected = [lang for lang, pattern in patterns.items() if re.search(pattern, text)]
    return detected or ["en"]


def detect_urgency(text: str) -> str:
    lowered = text.lower()
    high = ["urgent", "emergency", "asap", "immediately", "critical", "danger"]
    medium = ["soon", "quick", "today", "problem", "issue"]
    if any(token in lowered for token in high):
        return "High"
    if any(token in lowered for token in medium):
        return "Medium"
    return "Low"


def build_default_dataset() -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for intent, data in SERVICE_INTENTS.items():
        keywords = data["keywords"]
        for keyword in keywords:
            for template in TEMPLATES:
                texts.append(template.format(keyword=keyword))
                labels.append(intent)
        texts.append(data["description"])
        labels.append(intent)
    return texts, labels


@dataclass
class EchoLangBundle:
    model_pipeline: Pipeline
    labels: List[str]
    service_intents: Dict[str, Dict[str, List[str] | str]]
    version: str = "2.0.0"


class EchoLangPipeline:
    def __init__(self, bundle: EchoLangBundle):
        self.bundle = bundle

    @classmethod
    def train_default(
        cls,
        random_state: int = 42,
        memory: Optional[Union[str, Path, Memory]] = None,
    ) -> Tuple["EchoLangPipeline", Dict[str, str]]:
        texts, labels = build_default_dataset()
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=random_state,
            stratify=labels,
        )

        pipeline_memory: Optional[Memory]
        if isinstance(memory, Memory):
            pipeline_memory = memory
        elif memory is None:
            cache_dir = Path(".cache") / "sklearn"
            cache_dir.mkdir(parents=True, exist_ok=True)
            pipeline_memory = Memory(location=str(cache_dir), verbose=0)
        else:
            cache_dir = Path(memory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            pipeline_memory = Memory(location=str(cache_dir), verbose=0)

        model_pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(max_iter=300, random_state=random_state),
                ),
            ],
            memory=pipeline_memory,
        )
        model_pipeline.fit(x_train, y_train)

        report = classification_report(y_test, model_pipeline.predict(x_test), output_dict=True)
        metrics = {
            "accuracy": f"{report.get('accuracy', 0.0):.4f}",
            "macro_f1": f"{report.get('macro avg', {}).get('f1-score', 0.0):.4f}",
            "weighted_f1": f"{report.get('weighted avg', {}).get('f1-score', 0.0):.4f}",
        }

        bundle = EchoLangBundle(
            model_pipeline=model_pipeline,
            labels=sorted(set(labels)),
            service_intents=SERVICE_INTENTS,
        )
        return cls(bundle), metrics

    def analyze_text(self, text: str) -> Dict[str, object]:
        normalized = normalize_text(text)
        probs = self.bundle.model_pipeline.predict_proba([normalized])[0]
        labels = [str(label) for label in self.bundle.model_pipeline.classes_]
        scored = sorted(zip(labels, probs), key=lambda item: item[1], reverse=True)

        top_intent, top_score = scored[0]
        top_matches = {label: float(score) for label, score in scored[:3]}

        return {
            "original_text": text,
            "normalized_text": normalized,
            "detected_languages": detect_script_languages(text),
            "intent": top_intent,
            "confidence": float(top_score),
            "urgency": detect_urgency(text),
            "top_matches": top_matches,
        }

    def save(self, output_path: str) -> None:
        with open(output_path, "wb") as handle:
            pickle.dump(self.bundle, handle)

    @staticmethod
    def load(bundle_path: str) -> "EchoLangPipeline":
        with open(bundle_path, "rb") as handle:
            bundle: EchoLangBundle = pickle.load(handle)
        return EchoLangPipeline(bundle)
