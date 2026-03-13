from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import importlib
import os
import pickle
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


INTENT_CATALOG: Dict[str, Dict[str, Any]] = {
    "Worker Directory Access": {
        "description": "Blue-collar worker discovery, hiring, and service booking.",
        "action_template": [
            "Extract required trade, budget, and location from the request.",
            "Query yellow pages for matching workers within service radius.",
            "Return top providers with phone, rating, and ETA.",
        ],
        "keywords": {
            "en": [
                "plumber",
                "electrician",
                "carpenter",
                "cleaning",
                "repair",
                "driver",
                "mechanic",
            ],
            "hi": ["plumber", "bijli", "mistri", "safai", "repair", "driver"],
            "ta": ["plumber", "wire", "repair", "clean", "driver", "mechanic"],
        },
    },
    "Medical Conversation Support": {
        "description": "Doctor-patient dialog understanding and prescription drafting support.",
        "action_template": [
            "Extract patient complaints, duration, and symptom severity.",
            "Structure medication details: drug, dosage, frequency, duration.",
            "Generate a draft clinical summary for doctor validation.",
        ],
        "keywords": {
            "en": [
                "doctor",
                "hospital",
                "medicine",
                "prescription",
                "fever",
                "pain",
                "checkup",
            ],
            "hi": ["doctor", "dawai", "bukhar", "dard", "hospital", "parchi"],
            "ta": ["doctor", "marundhu", "fever", "vali", "hospital", "checkup"],
        },
    },
    "Emergency Assistance": {
        "description": "Critical incidents requiring immediate response and escalation.",
        "action_template": [
            "Collect exact location and emergency type immediately.",
            "Escalate to nearest emergency service channel.",
            "Provide caller with clear next-step safety instructions.",
        ],
        "keywords": {
            "en": ["emergency", "ambulance", "accident", "fire", "urgent", "police"],
            "hi": ["emergency", "ambulance", "hadsa", "aag", "urgent", "police"],
            "ta": ["emergency", "ambulance", "accident", "thee", "urgent", "police"],
        },
    },
}

HIGH_URGENCY_TOKENS = {
    "urgent",
    "emergency",
    "immediately",
    "critical",
    "danger",
    "ambulance",
    "fire",
    "accident",
    "police",
}

MEDIUM_URGENCY_TOKENS = {
    "soon",
    "quick",
    "today",
    "problem",
    "issue",
    "pain",
    "fever",
}

SCRIPT_PATTERNS = {
    "hi": re.compile(r"[\u0900-\u097F]"),
    "ta": re.compile(r"[\u0B80-\u0BFF]"),
    "en": re.compile(r"[A-Za-z]"),
}

NLLB_LANGUAGE_MAP = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "en": "eng_Latn",
}

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SUPPORTED_LANGUAGE_VARIANTS = ("en", "hi", "ta", "hinglish", "tanglish")

ROMAN_HI_TOKENS = {
    "mujhe",
    "chahiye",
    "jaldi",
    "dawai",
    "doctor",
    "aag",
    "hadsa",
    "madad",
}

ROMAN_TA_TOKENS = {
    "enakku",
    "venum",
    "marundhu",
    "irukku",
    "pannunga",
    "thee",
    "udhavi",
    "avasa",
}

STT_DOMAIN_PROMPT = (
    "This is Indian multilingual customer support audio with words like plumber, "
    "electrician, ambulance, emergency, doctor, hospital, medicine, marundhu, venum, "
    "mujhe, chahiye."
)

HI_TO_LATIN = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ट": "t", "ठ": "th",
    "ड": "d", "ढ": "dh", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b",
    "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "स": "s", "ह": "h",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n",
}

TA_TO_LATIN = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u", "ஊ": "uu", "எ": "e", "ஏ": "ee", "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au",
    "க": "ka", "ங": "nga", "ச": "sa", "ஜ": "ja", "ஞ": "nya", "ட": "ta", "ண": "na", "த": "tha", "ந": "na", "ப": "pa",
    "ம": "ma", "ய": "ya", "ர": "ra", "ல": "la", "வ": "va", "ழ": "zha", "ள": "la", "ற": "ra", "ன": "na", "ஹ": "ha",
    "ா": "a", "ி": "i", "ீ": "ii", "ு": "u", "ூ": "uu", "ெ": "e", "ே": "ee", "ை": "ai", "ொ": "o", "ோ": "oo", "ௌ": "au", "்": "",
}


@dataclass
class EchoLangBundle:
    version: str
    created_at: str
    backend: str
    labels: List[str]
    intent_catalog: Dict[str, Dict[str, Any]]
    classifier: Any
    tfidf_vectorizer: Optional[TfidfVectorizer]
    embed_model_name: Optional[str]
    training_metrics: Dict[str, str]
    language_vectorizer: Optional[TfidfVectorizer] = None
    language_classifier: Any = None


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def detect_script_mix(text: str) -> List[str]:
    detected: List[str] = []
    for lang, pattern in SCRIPT_PATTERNS.items():
        if pattern.search(text):
            detected.append(lang)
    return detected or ["en"]


def detect_urgency(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in HIGH_URGENCY_TOKENS):
        return "High"
    if any(token in lowered for token in MEDIUM_URGENCY_TOKENS):
        return "Medium"
    return "Low"


def _transliterate(text: str, mapping: Dict[str, str]) -> str:
    if not text:
        return ""
    out: List[str] = []
    for char in text:
        if ord(char) < 128:
            out.append(char)
        else:
            out.append(mapping.get(char, ""))
    return normalize_text("".join(out))


def _transliterate_hi_to_latin(text: str) -> str:
    return _transliterate(text, HI_TO_LATIN)


def _transliterate_ta_to_latin(text: str) -> str:
    return _transliterate(text, TA_TO_LATIN)


def _load_fleurs_texts(max_samples_per_language: int = 1200) -> Dict[str, List[str]]:
    try:
        datasets = importlib.import_module("datasets")
        load_dataset = getattr(datasets, "load_dataset")
    except Exception:
        return {}

    config_map = {
        "en": "en_us",
        "hi": "hi_in",
        "ta": "ta_in",
    }

    payload: Dict[str, List[str]] = {}
    for language, config in config_map.items():
        try:
            split = load_dataset("google/fleurs", config, split="train")
            texts: List[str] = []
            for row in split:
                text = normalize_text(str(row.get("transcription", "")))
                if text:
                    texts.append(text)
                if len(texts) >= max_samples_per_language:
                    break
            payload[language] = texts
        except Exception:
            payload[language] = []
    return payload


def _keyword_assign_intent(text: str) -> Optional[str]:
    lowered = text.lower()
    best_label = None
    best_hits = 0
    for label, payload in INTENT_CATALOG.items():
        tokens: List[str] = []
        for language_tokens in payload["keywords"].values():
            tokens.extend(language_tokens)
        hit_count = sum(1 for token in tokens if token.lower() in lowered)
        if hit_count > best_hits:
            best_hits = hit_count
            best_label = label
    return best_label if best_hits > 0 else None


def _base_language_for_variant(language: str) -> str:
    if language == "hinglish":
        return "hi"
    if language == "tanglish":
        return "ta"
    return language


def _append_romanized_variant(
    language: str,
    text: str,
    label: str,
    texts: List[str],
    labels: List[str],
    stats: Dict[str, int],
) -> None:
    if language == "hi":
        roman = _transliterate_hi_to_latin(text)
        if roman:
            texts.append(roman)
            labels.append(label)
            stats["hinglish"] += 1
    elif language == "ta":
        roman = _transliterate_ta_to_latin(text)
        if roman:
            texts.append(roman)
            labels.append(label)
            stats["tanglish"] += 1


def _build_fleurs_intent_samples(max_samples_per_language: int = 800) -> Tuple[List[str], List[str], Dict[str, int]]:
    fleurs = _load_fleurs_texts(max_samples_per_language=max_samples_per_language)
    texts: List[str] = []
    labels: List[str] = []
    stats = {"en": 0, "hi": 0, "ta": 0, "hinglish": 0, "tanglish": 0}

    for language in ("en", "hi", "ta"):
        for text in fleurs.get(language, []):
            label = _keyword_assign_intent(text)
            if label is None:
                continue
            texts.append(text)
            labels.append(label)
            stats[language] += 1
            _append_romanized_variant(language, text, label, texts, labels, stats)

    return texts, labels, stats


def _generate_language_labeled_samples() -> List[Dict[str, str]]:
    templates = {
        "en": [
            "I need {keyword} service near me",
            "Please arrange {keyword} today",
            "Need help with {keyword} urgently",
        ],
        "hi": [
            "मुझे {keyword} चाहिए",
            "मुझे {keyword} की मदद चाहिए",
            "{keyword} के लिए जल्दी मदद करो",
        ],
        "ta": [
            "எனக்கு {keyword} வேணும்",
            "{keyword} உதவி வேணும்",
            "{keyword} க்கு உடனே உதவி பண்ணுங்க",
        ],
        "hinglish": [
            "Mujhe {keyword} chahiye",
            "{keyword} ke liye jaldi help chahiye",
            "Mujhe {keyword} service abhi chahiye",
        ],
        "tanglish": [
            "Enakku {keyword} venum",
            "{keyword} ku immediate help pannunga",
            "{keyword} problem irukku, help venum",
        ],
    }

    rows: List[Dict[str, str]] = []
    for label, payload in INTENT_CATALOG.items():
        for language in SUPPORTED_LANGUAGE_VARIANTS:
            base_lang = _base_language_for_variant(language)
            for keyword in payload["keywords"][base_lang]:
                for template in templates[language]:
                    rows.append(
                        {
                            "text": normalize_text(template.format(keyword=keyword)),
                            "label": label,
                            "language": language,
                        }
                    )
    return rows


def _simulate_stt_noise(text: str) -> str:
    cleaned = normalize_text(text.lower())
    cleaned = re.sub(r"\b(please|kindly|sir|madam|ji)\b", " ", cleaned)
    cleaned = re.sub(r"\b(um+|uh+|ah+)\b", " ", cleaned)
    cleaned = re.sub(r"([a-z])\1{2,}", r"\1\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text


def _calibration_error(y_true: List[str], y_prob: List[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0

    bucket_total = [0 for _ in range(bins)]
    bucket_conf = [0.0 for _ in range(bins)]
    bucket_acc = [0.0 for _ in range(bins)]

    for is_correct, confidence in zip(y_true, y_prob):
        idx = min(int(confidence * bins), bins - 1)
        bucket_total[idx] += 1
        bucket_conf[idx] += confidence
        bucket_acc[idx] += float(is_correct)

    total = float(len(y_true))
    ece = 0.0
    for idx in range(bins):
        if bucket_total[idx] == 0:
            continue
        acc = bucket_acc[idx] / bucket_total[idx]
        conf = bucket_conf[idx] / bucket_total[idx]
        ece += (bucket_total[idx] / total) * abs(acc - conf)
    return ece


def _try_train_language_identifier(random_state: int = 42) -> Tuple[Optional[TfidfVectorizer], Any, Dict[str, str]]:
    samples: List[str] = []
    labels: List[str] = []
    source = "heuristic"

    seed_examples = {
        "en": ["need doctor urgently", "please send ambulance", "find electrician near me"],
        "hi": ["मुझे डॉक्टर चाहिए", "एम्बुलेंस तुरंत भेजो", "मुझे प्लंबर की मदद चाहिए"],
        "ta": ["எனக்கு டாக்டர் வேணும்", "ஆம்புலன்ஸ் உடனே வேண்டும்", "எலக்ட்ரீஷன் உதவி வேணும்"],
        "hinglish": ["mujhe doctor chahiye", "ambulance jaldi bhejo", "plumber ki help chahiye"],
        "tanglish": ["enakku doctor venum", "ambulance odane venum", "plumber help pannunga"],
    }

    for label, rows in seed_examples.items():
        for row in rows:
            samples.append(row)
            labels.append(label)

    fleurs = _load_fleurs_texts(max_samples_per_language=1200)
    if any(fleurs.values()):
        source = "google/fleurs"

    for row in fleurs.get("en", []):
        samples.append(row)
        labels.append("en")
    for row in fleurs.get("hi", []):
        samples.append(row)
        labels.append("hi")
        roman = _transliterate_hi_to_latin(row)
        if roman:
            samples.append(roman)
            labels.append("hinglish")
    for row in fleurs.get("ta", []):
        samples.append(row)
        labels.append("ta")
        roman = _transliterate_ta_to_latin(row)
        if roman:
            samples.append(roman)
            labels.append("tanglish")

    if len(set(labels)) < 3:
        return None, None, {"language_identifier_source": "fallback-heuristic"}

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    x = vectorizer.fit_transform(samples)
    classifier = LogisticRegression(max_iter=400, random_state=random_state)
    classifier.fit(x, labels)

    return vectorizer, classifier, {
        "language_identifier_source": source,
        "language_identifier_samples": str(len(samples)),
    }


def _predict_language_variant_with_model(
    text: str,
    vectorizer: Optional[TfidfVectorizer],
    classifier: Any,
) -> Optional[str]:
    if vectorizer is None or classifier is None:
        return None
    try:
        features = vectorizer.transform([text])
        predicted = str(classifier.predict(features)[0])
        if predicted in SUPPORTED_LANGUAGE_VARIANTS:
            return predicted
    except Exception:
        return None
    return None


def _detect_roman_mixed_variant(text: str) -> str:
    lowered = text.lower()
    hi_count = sum(1 for token in ROMAN_HI_TOKENS if token in lowered)
    ta_count = sum(1 for token in ROMAN_TA_TOKENS if token in lowered)
    if hi_count > ta_count and hi_count > 0:
        return "hinglish"
    if ta_count > hi_count and ta_count > 0:
        return "tanglish"
    return "en"


def _romanized_eval_rows(language: str, text: str, label: str) -> List[Dict[str, str]]:
    if language == "hi":
        roman = _transliterate_hi_to_latin(text)
        if roman:
            return [{"text": roman, "label": label, "language": "hinglish"}]
        return []
    if language == "ta":
        roman = _transliterate_ta_to_latin(text)
        if roman:
            return [{"text": roman, "label": label, "language": "tanglish"}]
        return []
    return []


def _append_fleurs_eval_rows(rows: List[Dict[str, str]], max_fleurs_per_language: int) -> None:
    fleurs = _load_fleurs_texts(max_samples_per_language=max_fleurs_per_language)
    for language in ("en", "hi", "ta"):
        for text in fleurs.get(language, []):
            label = _keyword_assign_intent(text)
            if label is None:
                continue
            rows.append({"text": text, "label": label, "language": language})
            rows.extend(_romanized_eval_rows(language, text, label))


def _build_eval_slices(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    slices: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        text_mode_key = (row["language"], "text")
        audio_mode_key = (row["language"], "audio_simulated")
        slices.setdefault(text_mode_key, []).append(
            {"text": row["text"], "label": row["label"]}
        )
        slices.setdefault(audio_mode_key, []).append(
            {"text": _simulate_stt_noise(row["text"]), "label": row["label"]}
        )
    return slices


def _score_eval_subset(
    pipeline: "EchoLangPipeline",
    subset: List[Dict[str, str]],
) -> Dict[str, Any]:
    y_true: List[str] = []
    y_pred: List[str] = []
    y_conf: List[float] = []

    for item in subset:
        prediction = pipeline.analyze_text(item["text"])
        y_true.append(item["label"])
        y_pred.append(str(prediction["intent"]))
        y_conf.append(float(prediction["confidence"]))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    correct_flags = [truth == pred for truth, pred in zip(y_true, y_pred)]

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_conf": y_conf,
        "correct_flags": correct_flags,
        "accuracy": round(float(report.get("accuracy", 0.0)), 4),
        "macro_f1": round(float(report.get("macro avg", {}).get("f1-score", 0.0)), 4),
        "ece": round(float(_calibration_error(correct_flags, y_conf)), 4),
    }


def _generate_training_samples() -> Tuple[List[str], List[str]]:
    templates = [
        "I need {keyword} service near me",
        "Please arrange {keyword} today",
        "Need help with {keyword} urgently",
        "Can you support {keyword} request",
        "Mujhe {keyword} chahiye",
        "Mujhe {keyword} service jaldi chahiye",
        "Enakku {keyword} venum",
        "Enakku {keyword} help venum now",
        "{keyword} issue hai, please help",
        "{keyword} problem irukku, help pannunga",
    ]

    texts: List[str] = []
    labels: List[str] = []

    for label, payload in INTENT_CATALOG.items():
        texts.append(payload["description"])
        labels.append(label)
        for language in ("en", "hi", "ta"):
            for keyword in payload["keywords"][language]:
                for template in templates:
                    texts.append(template.format(keyword=keyword))
                    labels.append(label)

    return texts, labels


class EchoLangPipeline:
    def __init__(
        self,
        bundle: EchoLangBundle,
        whisper_model_name: str = "small",
        mt_model_name: str = "facebook/nllb-200-distilled-600M",
    ) -> None:
        self.bundle = bundle
        self.whisper_model_name = whisper_model_name
        self.mt_model_name = mt_model_name
        self._embedder = None
        self._translator = None
        self._whisper_model = None

    @classmethod
    def train_default(
        cls,
        random_state: int = 42,
        prefer_sbert: bool = True,
        use_fleurs_dataset: bool = True,
    ) -> Tuple["EchoLangPipeline", Dict[str, str]]:
        texts, labels = _generate_training_samples()

        fleurs_stats: Dict[str, int] = {}
        if use_fleurs_dataset:
            fleurs_texts, fleurs_labels, fleurs_stats = _build_fleurs_intent_samples(
                max_samples_per_language=800
            )
            texts.extend(fleurs_texts)
            labels.extend(fleurs_labels)

        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=random_state,
            stratify=labels,
        )

        backend = "tfidf"
        tfidf_vectorizer: Optional[TfidfVectorizer] = None
        embed_model_name: Optional[str] = None

        if prefer_sbert:
            try:
                embed_model_name = DEFAULT_EMBED_MODEL
                sentence_transformers = importlib.import_module("sentence_transformers")
                sentence_transformer_cls = getattr(sentence_transformers, "SentenceTransformer")
                embedder = sentence_transformer_cls(embed_model_name)
                x_train_features = embedder.encode(x_train, normalize_embeddings=True)
                x_test_features = embedder.encode(x_test, normalize_embeddings=True)
                backend = "sbert"
            except Exception:
                prefer_sbert = False

        if not prefer_sbert:
            tfidf_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            x_train_features = tfidf_vectorizer.fit_transform(x_train)
            x_test_features = tfidf_vectorizer.transform(x_test)

        classifier = LogisticRegression(max_iter=500, random_state=random_state)
        classifier.fit(x_train_features, y_train)

        language_vectorizer, language_classifier, lid_metrics = _try_train_language_identifier(
            random_state=random_state
        )

        report = classification_report(y_test, classifier.predict(x_test_features), output_dict=True)
        metrics = {
            "accuracy": f"{report.get('accuracy', 0.0):.4f}",
            "macro_f1": f"{report.get('macro avg', {}).get('f1-score', 0.0):.4f}",
            "weighted_f1": f"{report.get('weighted avg', {}).get('f1-score', 0.0):.4f}",
            "backend": backend,
            "samples": str(len(texts)),
        }
        metrics.update(lid_metrics)
        if fleurs_stats:
            metrics["fleurs_weak_label_total"] = str(sum(fleurs_stats.values()))
            metrics["fleurs_hinglish_samples"] = str(fleurs_stats.get("hinglish", 0))
            metrics["fleurs_tanglish_samples"] = str(fleurs_stats.get("tanglish", 0))

        bundle = EchoLangBundle(
            version="3.0.0",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            backend=backend,
            labels=sorted(set(labels)),
            intent_catalog=INTENT_CATALOG,
            classifier=classifier,
            tfidf_vectorizer=tfidf_vectorizer,
            embed_model_name=embed_model_name,
            training_metrics=metrics,
            language_vectorizer=language_vectorizer,
            language_classifier=language_classifier,
        )
        return cls(bundle=bundle), metrics

    def save(self, output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("wb") as handle:
            pickle.dump(self.bundle, handle)

    @staticmethod
    def load(bundle_path: str) -> "EchoLangPipeline":
        with Path(bundle_path).open("rb") as handle:
            bundle: EchoLangBundle = pickle.load(handle)
        return EchoLangPipeline(bundle=bundle)

    def _lazy_embedder(self):
        if self._embedder is None:
            if self.bundle.embed_model_name is None:
                raise RuntimeError("SBERT backend requested but embed model is not configured.")
            sentence_transformers = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(sentence_transformers, "SentenceTransformer")
            self._embedder = sentence_transformer_cls(self.bundle.embed_model_name)
        return self._embedder

    def _lazy_translator(self):
        if self._translator is None:
            transformers_module = importlib.import_module("transformers")
            pipeline = getattr(transformers_module, "pipeline")
            self._translator = pipeline(
                "translation",
                model=self.mt_model_name,
                device=-1,
            )
        return self._translator

    def _lazy_whisper(self):
        if self._whisper_model is None:
            whisper_module = importlib.import_module("whisper")
            self._whisper_model = whisper_module.load_model(self.whisper_model_name)
        return self._whisper_model

    def _detect_language_hint(self, text: str) -> str:
        variant = self.detect_language_variant(text)
        if variant in ("hi", "ta"):
            return variant
        return "en"

    def detect_language_variant(self, text: str) -> str:
        normalized = normalize_text(text)
        vectorizer = getattr(self.bundle, "language_vectorizer", None)
        classifier = getattr(self.bundle, "language_classifier", None)

        from_model = _predict_language_variant_with_model(normalized, vectorizer, classifier)
        if from_model is not None:
            return from_model

        scripts = detect_script_mix(text)
        if "hi" in scripts and "ta" not in scripts:
            return "hi"
        if "ta" in scripts and "hi" not in scripts:
            return "ta"
        if scripts == ["en"]:
            return _detect_roman_mixed_variant(normalized)

        return _detect_roman_mixed_variant(normalized)

    def _preprocess_audio_bytes(self, audio_bytes: bytes) -> bytes:
        try:
            soundfile = importlib.import_module("soundfile")
            sf_read = getattr(soundfile, "read")
            sf_write = getattr(soundfile, "write")
            data, sample_rate = sf_read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
            waveform = np.mean(data, axis=1)
            peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
            if peak > 0:
                waveform = waveform / max(peak, 1e-6)

            target_sr = 16000
            if sample_rate != target_sr and waveform.size > 1:
                # Lightweight resampling without extra dependencies.
                x_old = np.linspace(0.0, 1.0, num=len(waveform), endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=int(len(waveform) * target_sr / sample_rate), endpoint=False)
                waveform = np.interp(x_new, x_old, waveform).astype(np.float32)
                sample_rate = target_sr

            with io.BytesIO() as output:
                sf_write(output, waveform, sample_rate, format="WAV")
                return output.getvalue()
        except Exception:
            return audio_bytes

    def _clean_transcript(self, text: str) -> str:
        cleaned = normalize_text(text)
        cleaned = re.sub(r"\b(uh+|um+|ah+|hmm+)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def transcribe_audio_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
        model = self._lazy_whisper()
        processed_audio = self._preprocess_audio_bytes(audio_bytes)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(processed_audio)
            temp_path = temp.name

        try:
            primary = model.transcribe(
                temp_path,
                fp16=False,
                task="transcribe",
                condition_on_previous_text=False,
                no_speech_threshold=0.5,
                logprob_threshold=-1.0,
                temperature=0.0,
                initial_prompt=STT_DOMAIN_PROMPT,
            )
            secondary = model.transcribe(
                temp_path,
                fp16=False,
                task="transcribe",
                condition_on_previous_text=False,
                no_speech_threshold=0.45,
                logprob_threshold=-1.2,
                temperature=(0.0, 0.2, 0.4),
            )

            primary_text = self._clean_transcript(str(primary.get("text", "")))
            secondary_text = self._clean_transcript(str(secondary.get("text", "")))
            primary_quality = sum(ch.isalnum() for ch in primary_text)
            secondary_quality = sum(ch.isalnum() for ch in secondary_text)

            best_text = primary_text if primary_quality >= secondary_quality else secondary_text
            best_language = (
                primary.get("language", "unknown")
                if primary_quality >= secondary_quality
                else secondary.get("language", "unknown")
            )

            return {
                "text": best_text,
                "detected_language": best_language,
                "stt_quality_chars": max(primary_quality, secondary_quality),
            }
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def translate_to_english(self, text: str, language_hint: Optional[str] = None) -> str:
        language = language_hint or self._detect_language_hint(text)
        if language == "en":
            return text

        translator = self._lazy_translator()
        src_lang = NLLB_LANGUAGE_MAP.get(language, "eng_Latn")
        try:
            translated = translator(
                text,
                src_lang=src_lang,
                tgt_lang="eng_Latn",
                max_length=512,
            )
            return normalize_text(translated[0]["translation_text"])
        except Exception:
            # If translation fails, keep original text to avoid full pipeline failure.
            return text

    def _predict_scores(self, text_en: str) -> List[Tuple[str, float]]:
        if self.bundle.backend == "sbert":
            embedder = self._lazy_embedder()
            features = embedder.encode([text_en], normalize_embeddings=True)
            probs = self.bundle.classifier.predict_proba(features)[0]
        else:
            if self.bundle.tfidf_vectorizer is None:
                raise RuntimeError("TF-IDF backend selected but vectorizer is missing from bundle.")
            features = self.bundle.tfidf_vectorizer.transform([text_en])
            probs = self.bundle.classifier.predict_proba(features)[0]

        labels = [str(label) for label in self.bundle.classifier.classes_]
        scored = sorted(zip(labels, probs), key=lambda row: row[1], reverse=True)
        return [(label, float(score)) for label, score in scored]

    def analyze_text(self, text: str) -> Dict[str, Any]:
        original = normalize_text(text)
        lang_variant = self.detect_language_variant(original)
        lang_hint = self._detect_language_hint(original)
        translated = self.translate_to_english(original, language_hint=lang_hint)

        scored = self._predict_scores(translated)
        top_label, top_score = scored[0]
        top3 = scored[:3]

        actionable = {
            label: self.bundle.intent_catalog[label]["action_template"] for label, _ in top3
        }

        return {
            "input_mode": "text",
            "original_text": original,
            "language_variant": lang_variant,
            "language_hint": lang_hint,
            "translated_text": translated,
            "intent": top_label,
            "confidence": top_score,
            "urgency": detect_urgency(original),
            "top_intents": [
                {"intent": label, "percentage": round(score * 100.0, 2)} for label, score in top3
            ],
            "actionable": actionable,
        }

    def analyze_audio(self, audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
        stt = self.transcribe_audio_bytes(audio_bytes=audio_bytes, suffix=suffix)
        analysis = self.analyze_text(stt["text"])
        analysis["input_mode"] = "audio"
        analysis["stt_detected_language"] = stt["detected_language"]
        analysis["stt_quality_chars"] = stt.get("stt_quality_chars", 0)
        return analysis

    def evaluate_sliced_performance(
        self,
        include_fleurs: bool = True,
        max_fleurs_per_language: int = 300,
    ) -> Dict[str, Any]:
        rows = _generate_language_labeled_samples()

        if include_fleurs:
            _append_fleurs_eval_rows(rows, max_fleurs_per_language)

        slices = _build_eval_slices(rows)

        results: Dict[str, Any] = {
            "overall": {},
            "by_language_mode": {},
        }

        all_true: List[str] = []
        all_pred: List[str] = []
        all_correct: List[bool] = []
        all_conf: List[float] = []

        for (language, mode), subset in slices.items():
            scored = _score_eval_subset(self, subset)

            results["by_language_mode"][f"{language}:{mode}"] = {
                "samples": len(subset),
                "accuracy": scored["accuracy"],
                "macro_f1": scored["macro_f1"],
                "ece": scored["ece"],
            }

            all_true.extend(scored["y_true"])
            all_pred.extend(scored["y_pred"])
            all_correct.extend(scored["correct_flags"])
            all_conf.extend(scored["y_conf"])

        overall_report = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
        results["overall"] = {
            "samples": len(all_true),
            "accuracy": round(float(overall_report.get("accuracy", 0.0)), 4),
            "macro_f1": round(float(overall_report.get("macro avg", {}).get("f1-score", 0.0)), 4),
            "ece": round(float(_calibration_error(all_correct, all_conf)), 4),
        }

        return results

    def model_card(self) -> Dict[str, Any]:
        return {
            "bundle": asdict(self.bundle),
            "runtime": {
                "whisper_model": self.whisper_model_name,
                "translation_model": self.mt_model_name,
            },
        }
