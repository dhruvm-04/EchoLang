# How to Run EchoLang

## 1. Prerequisites

- Python 3.9 or higher (tested with Python 3.11)
- PyTorch (with CUDA for GPU acceleration is recommended)
- Adequate RAM and disk space
- Optional: Jupyter Notebook or JupyterLab for interactive usage

## 2. Installation

### a) Clone the Repository
```git clone https://github.com/dhruvm-04/EchoLang.git```  
```cd EchoLang```

### b) Set Up a Virtual Environment (Recommended)

```python -m venv .venv```  
```source .venv/bin/activate # On Windows: .venv\Scripts\activate```


### c) Install Dependencies

```pip install -r requirements.txt```
- If you face issues with PyTorch installation, refer to the official PyTorch site for instructions based on your system configuration.

## 3. Running EchoLang

### a) Via Jupyter Notebook

From the project root:

```jupyter notebook EchoLang.ipynb```

- Choose the Python kernel corresponding to your virtual environment.
- Execute each cell in sequence; sample audio and text examples are provided.

### b) Python Script (Optional)

You may adapt the codebase as a script for backend/API use. Import components from `src/` or reference code cells in the provided notebook.

## 4. Usage

- **Text Input:** Enter or paste multilingual user requests.
- **Audio Input:** Upload or record `.wav` audio files (Hindi, Tamil, English, or code-mixed).
- **Output:** The system will detect language, transcribe, optionally translate, and classify service intent and urgency.

## 5. Notes

- For optimal accuracy, use clear `.wav` audio at 16kHz or higher.
- Model files (Whisper, NLLB-200) may be large and require time to download at first use.
- GPU acceleration is advised for faster processing, but CPU mode is also supported.

## 6. Troubleshooting

- If installing dependencies fails, ensure PyTorch is installed, then rerun `pip install -r requirements.txt`.
- For audio handling issues, check that `librosa` and `soundfile` are correctly installed and audio files are `.wav` format.

---

For more detailed help, see the `README.md` file in the repository or open an issue on the project's page.
