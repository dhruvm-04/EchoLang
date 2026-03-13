# EchoLang - Real-Time Multilingual Speech Translator

EchoLang is an end-to-end multilingual pipeline for:
- Speech or text input in `English`, `Hindi`, `Hinglish`, `Tamil`, or `Tamil+English`
- STT when audio is provided
- Translation to English (only when needed)
- 3-intent service analysis with probability percentages
- Actionable next-step routing for downstream systems

## Problem Focus
The system is optimized for Tier 2/3 India use cases:
- Yellow-page style blue-collar worker discovery and booking
- Doctor-patient conversation support for medical note/prescription drafting
- Emergency support routing

## Rebuilt Architecture
1. Input: text or speech
2. STT (audio path): Whisper
3. Language hint + translation to English: NLLB-200 (Hindi/Tamil when required)
4. Intent analysis: trained classifier stored in `.pkl`
5. Output: top 3 intent percentages + actionable steps

## Three Main Intents
1. `Worker Directory Access`
2. `Medical Conversation Support`
3. `Emergency Assistance`

## Artifact Contract
Kaggle training exports one artifact:
- `echolang_bundle.pkl`

This artifact includes:
- Trained intent model
- Label space and intent metadata
- Action templates
- Training metrics and version metadata

The Streamlit app loads this artifact directly.

## Project Structure
```text
src/echolang/pipeline.py   # Training + runtime (STT/MT/Intent)
src/echolang/cli.py        # CLI inference for text/audio
kaggle_build_artifact.py   # Kaggle export entry point
streamlit_app.py           # Professional UI consuming .pkl
EchoLang.ipynb             # Kaggle-style notebook workflow
```

## Local Run
```bash
pip install -r requirements.txt
python kaggle_build_artifact.py --output echolang_bundle.pkl
streamlit run streamlit_app.py
```

## Kaggle Run
```bash
pip install -r requirements.txt
python kaggle_build_artifact.py --output /kaggle/working/echolang_bundle.pkl
```

## CLI Usage
```bash
python -m src.echolang.cli --model echolang_bundle.pkl --text "Mujhe plumber chahiye urgently"
python -m src.echolang.cli --model echolang_bundle.pkl --audio sample.wav
```

## Notes
- If SBERT is available, training uses multilingual sentence embeddings.
- If SBERT is unavailable, training falls back to TF-IDF + Logistic Regression.
- Audio model weights are loaded at runtime; the `.pkl` stores the trained analyzer artifact.
