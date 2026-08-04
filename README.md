# EchoLang

EchoLang is a small FastAPI app for handling service requests in mixed Hindi, Tamil, and English. It can take text or audio, transcribe speech, translate it, and classify the request by intent and urgency.

## What it does

- Transcribe audio into text
- Analyze text for language, translation, intent, confidence, and urgency
- Return structured JSON that can be used by a UI or another service

## Run locally

```bash
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload
```

Create a .env file with Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Then open http://127.0.0.1:8000/docs.

## Main endpoints

- GET /health
- POST /transcribe
- POST /analyze
- POST /process

## Frontend

A simple browser UI is included in the frontend folder. You can serve it with:

```bash
cd frontend
python -m http.server 8899
```

Then open http://127.0.0.1:8899.

## Project files

- main.py — FastAPI app and request handling
- prompts.py — prompt setup for the model
- schemas.py — request and response models
- config.py — config and category list
- frontend/ — static UI
=======
# EchoLang

EchoLang is a small FastAPI app for handling service requests in mixed Hindi, Tamil, and English. It can take text or audio, transcribe speech, translate it, and classify the request by intent and urgency.

## What it does

- Transcribe audio into text
- Analyze text for language, translation, intent, confidence, and urgency
- Return structured JSON that can be used by a UI or another service

## Run locally

```bash
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload
```

Create a .env file with Groq API key:

```env
GROQ_API_KEY=groq_api_key_here
```

Then open http://127.0.0.1:8000/docs.

## Main endpoints

- GET /health
- POST /transcribe
- POST /analyze
- POST /process

## Frontend

A simple browser UI is included in the frontend folder. You can serve it with:

```bash
cd frontend
python -m http.server 8001
```

Then open http://127.0.0.1:8001.

## Project files

- main.py — FastAPI app and request handling
- prompts.py — prompt setup for the model
- schemas.py — request and response models
- config.py — config and category list
- frontend/ — static UI
>>>>>>> 61154f6341ee654bd05c573adaf8d1c775b8f99c
