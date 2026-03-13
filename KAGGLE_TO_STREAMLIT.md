# Kaggle to Streamlit Workflow

## 1. Build model bundle in Kaggle

If you uploaded the project as a Kaggle Dataset, first inspect the mounted input path:

```bash
!find /kaggle/input -maxdepth 2 -type d
```

Then copy the project into the writable working directory using the real dataset path, for example:

```bash
!cp -r /kaggle/input/echolang/EchoLang /kaggle/working/EchoLang
%cd /kaggle/working/EchoLang
```

If your dataset path is different, replace `/kaggle/input/echolang/EchoLang` with the path shown by `find`.

From the project root, run:

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
