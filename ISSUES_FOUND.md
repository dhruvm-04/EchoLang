# EchoLang Audit - Key Issues

## Critical Runtime Issues

1. `EchoLang.ipynb` uses undefined variables in evaluation cells:
- `TEST_SPLIT`, `MAX_SAMPLES`, `processor`, `model`, `wer_metric`, `cer_metric` are referenced before guaranteed initialization.

2. The notebook monkey-patches global torch functions:
- Reassigning `torch.tensor` and `torch.from_numpy` is unsafe and can break unrelated code paths.

3. ASR language is hard-forced to English during transcription:
- In `_transcribe_with_direct_fix`, `language='en'` reduces multilingual recognition quality.

4. N-gram model is never actually persisted in `_ngram_models`:
- `load_ngram_model` and `_create_simple_ngram_model` print success messages but do not build usable language model objects.

## Structural Issues

1. Documentation claims a `src/` architecture but repo was notebook-only.
2. UI, model loading, preprocessing, and evaluation are tightly coupled in one notebook.
3. No reproducible artifact-export step for deployment.
4. Requirements list is heavy and not deployment-targeted for a simple Streamlit host.

## Deployment Gap

- No single-file model artifact was exported for Streamlit consumption. This blocks easy hosting and sharing.
