# Kaggle to Streamlit Workflow

## 1. Build model bundle in Kaggle

Run these commands in a Kaggle notebook cell from your project root:

```bash
pip install -r requirements.txt
python kaggle_build_artifact.py --output /kaggle/working/echolang_bundle.pkl
```

You should see metrics and a generated file at:
- `/kaggle/working/echolang_bundle.pkl`

## 2. Download artifact from Kaggle

In the right sidebar of Kaggle, open the `Output` section and download:
- `echolang_bundle.pkl`

## 3. Use in Streamlit

Place the file beside `streamlit_app.py` and run:

```bash
streamlit run streamlit_app.py
```

If you keep the file in another location, set the path inside the app input.

## 4. Optional sanity test

```bash
python -c "from src.echolang.pipeline import EchoLangPipeline; p=EchoLangPipeline.load('echolang_bundle.pkl'); print(p.analyze_text('Need urgent plumber help'))"
```
