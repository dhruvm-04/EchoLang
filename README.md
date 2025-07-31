# EchoLang

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-orange)

Real-time Speech-to-Text (STT) and translation system for code-switched Hindi-Tamil-English audio. Optimized for low-resource settings, it enables contextual transcription and intent-aware translation for use cases like voice-based search and medical transcription.

## Problem Statement
Real time speech to text (STT) translation and transcription tool, focused on a mix of Hindi/Tamil with English. (Transcription and translation of mixed Hindi-English (Hinglish) and Tamil-English speech into English text.)
Considering the use case - Accessing a Yellow Page Directory for blue collar workers living in Tier 2/3 cities of India, that can speak a mix of **English/Hindi/Tamil**

## Overview

EchoLang is an intelligent multilingual service request analysis system designed to process and categorize service requests in **Hindi**, **Tamil**, and **English**. The system combines advanced speech recognition, neural machine translation, and intent classification to provide a comprehensive solution for service request management.

## Key Features

- Speech-to-Text: Advanced ASR using Whisper with n-gram language model fusion
- Multilingual Support: Native support for Hindi, Tamil, and English
- Neural Translation: NLLB-200 powered translation for cross-language understanding
- Intent Classification: Automated service category detection with confidence scoring
- Interactive Chatbot: Conversational interface for guided service request collection
- Real-time Analysis: Instant processing with detailed confidence metrics
- Service Categories: Emergency Services, Healthcare, Home Maintenance, Transportation, Cleaning, and General Services

## Prerequisites
- torch>=2.0.0 transformers>=4.30.0 openai-whisper>=20231117
- sentence-transformers>=2.2.0 langdetect>=1.0.9 jiwer>=3.0.0
- soundfile>=0.12.0 librosa>=0.10.0 accelerate>=0.20.0

## Project Structure
  EchoLang/  
  ├── docs/ # Documentation  
  ├── src/ # Source code  
  │ ├── models/ # Model management  
  │ ├── processing/ # Audio/text processing  
  │ ├── translation/ # Translation components  
  │ ├── classification/ # Intent classification  
  │ └── ui/ # User interface  
  ├── tests/ # Unit tests  
  ├── examples/ # Usage examples  
  └── notebooks/ # Jupyter notebooks  

## Core Features

1. **Automatic Speech Recognition (ASR)**
   - **Tool:** OpenAI Whisper  
   - *Transcribes* code-switched speech (Hindi/Tamil + English) in **real time**, robust to noisy and low-resource audio.
   - **Resamples** audio from 32kHz to 16kHz using SciPy to meet Whisper’s input requirements.

2. **Machine Translation (MT)**
   - **Tool:** Meta NLLB-200  
   - *Translates* Hindi/Tamil fragments to fluent English after initial transcription.
   - Preserves context, named entities, and domain terminology, crucial for code-switched utterances.

3. **Intent Classification and Semantic Understanding**
   - **Tool:** Sentence-BERT (all-MiniLM-L6-v2)
   - Encodes *user queries* and *service categories* into vectors and matches intent using **cosine similarity**, allowing zero-shot classification and support for diverse user phrasing and dialects.

4. **Chatbot**
   - **Natively developed** with a rule-based logic for handling a variety of input types and languages, optimized for multilingual, low-literacy users.
   - Workflow:
     - Input type detection (text/voice)
     - Language detection (Hindi/Tamil/English)
     - Transcription & translation (Whisper + NLLB-200) and intent classification (Sentence-BERT)
     - Urgency recognition via keyword rules
     - Confirmation flow for user feedback and corrections

5. **Model Manager Functions**
   - **Natively developed** functions:
     - `_load_whisper()`: Loads Whisper for ASR (16 kHz mono audio, mixed-language)
     - `_load_translator()`: Starts NLLB-200 for translation to English
     - `_load_intent_model()`: Loads Sentence-BERT for embeddings
     - `_compute_intent_embeddings()`: Pre-encodes service categories for runtime matching
     - `apply_ngram_fusion()`: Applies n-gram, rule-based corrections for transcription errors (e.g., mapping "plambing" to "plumbing").

---

## Chatbot

- **Architecture:** Rule-based, focused on clarity for code-switched, multilingual, and accessible interactions.
- **Flow:**
  - *Detects* if input is text or voice and identifies Hindi, Tamil, or English using Unicode-based rules.
  - *Transcribes* and *translates* speech with Whisper and NLLB-200.
  - *Classifies intent* (Sentence-BERT) and *detects urgency* (rule-based keywords).
  - *Engages* user with confirmation and allows corrections before simulating the requested action.

---

## Model Manager Functions

- **_load_whisper()**: Loads the Whisper model for audio-to-text.
- **_load_translator()**: Sets up NLLB-200 for Hindi/Tamil to English translation.
- **_load_intent_model()**: Loads Sentence-BERT model for processing text into embeddings.
- **_compute_intent_embeddings()**: Converts service categories to vector form for matching.
- **apply_ngram_fusion()**: Corrects common transcription mistakes using simple keyword rules.

---

## Pipeline Explanation

The cascaded modular architecture of EchoLang is intended to manage multilingual user inputs in Tier 2/3 Indian real-world settings. There are four main steps in the pipeline's processing of user speech:

**Audio Input and Preprocessing:**  
Wav files are required for audio inputs. To satisfy Whisper's specifications, all input waveforms are automatically resampled from 32kHz to 16kHz using `scipy.signal.resample`. After being transformed into a NumPy array, the waveform is sent straight to the transcription engine.

**Automatic Speech Recognition (ASR):**  
The audio is transcribed using OpenAI's Whisper (medium) model. It can handle background noise and dialect variation with high robustness and supports code-switched inputs in Tamil, Hindi, and English. Additionally, Whisper recognizes the most common spoken language without the use of explicit language tags.

**Postprocessing and N-gram Fusion:**  
After transcription is finished, frequent low-level substitution or spelling errors can be fixed with optional n-gram-based keyword correction. Using a pseudo dictionary, this helps refine terms like "elektrician" to "electrician." The final pipeline does not use neural translation.

**Intent Classification (SBERT):**  
Sentence-BERT (all-MiniLM-L6-v2) is used to embed the clean English text, and cosine similarity is calculated against a predetermined embedding bank of service categories. The intended service type is chosen from the category with the highest degree of similarity. Keyword matching is used in parallel to classify urgency (e.g., "immediately," "urgent").

This structure allows EchoLang to be robust with handling informal phrasing and multilingualism.

---

## Model Performance

- **Tested With:** Google FLEURS dataset (English, Hindi, Tamil)
- **ASR Metrics:**

| Language | Word Error Rate (WER) | Character Error Rate (CER) | Remark                          |
|----------|----------------------|----------------------------|----------------------------------|
| English  | 0.2488               | 0.0641                     | High transcription accuracy      |
| Hindi    | 0.4705               | 0.2237                     | Moderate, code-mixing challenges |
| Tamil    | 0.6478               | 0.2399                     | Errors, semantic info kept       |

- Lower-resource languages (Hindi, Tamil) exhibited **higher error rates** due to code-mixing and phonetic differences, but overall, the system retained substantial **semantic accuracy**.

---

## Tools and Libraries Used

- **ASR:** OpenAI Whisper (whisper-medium)
- **Translation:** Meta NLLB-200
- **Semantic Classification:** Sentence-BERT (all-MiniLM-L6-v2)
- **Core Frameworks:** Torch, PyTorch
- **Signal Processing/IO:** SciPy (`scipy.signal`), NumPy, torchaudio
- **Utilities:** regex, sentence-transformers
