import json
import tempfile
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq

from config import GROQ_API_KEY, STT_MODEL, LLM_MODEL, SERVICE_CATEGORIES
from prompts import build_messages
from schemas import AnalyzeRequest, AnalyzeResult, TranscribeResult, ProcessResult

app = FastAPI(
    title="EchoLang",
    description=(
        "Multilingual (Hindi/Tamil/English) service-request triage: "
        "speech-to-text + translation + intent classification + urgency detection, "
        "powered by Groq's free-tier API."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly: the static frontend can be opened from file:// or any host
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)


def _transcribe(file_bytes: bytes, filename: str) -> str:
    """Send audio to Groq's hosted Whisper for fast STT."""
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(filename, audio_file.read()),
                model=STT_MODEL,
                response_format="text",
            )
        return str(transcription).strip()
    finally:
        os.remove(tmp_path)


def _analyze(text: str) -> AnalyzeResult:
    """Run the prompt-engineered translation + intent + urgency pass."""
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=build_messages(text),
        temperature=0.2,  # low temperature: we want consistent, deterministic-ish JSON
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Model returned malformed JSON")

    # Guard against category hallucination despite prompt constraints
    if data.get("intent_category") not in SERVICE_CATEGORIES:
        data["intent_category"] = "General Services"
        data["confidence"] = min(float(data.get("confidence", 0.5)), 0.5)

    return AnalyzeResult(
        original_text=text,
        detected_language=data.get("detected_language", "Unknown"),
        translated_text=data.get("translated_text", text),
        intent_category=data["intent_category"],
        confidence=float(data.get("confidence", 0.5)),
        urgency=data.get("urgency", "low"),
        urgency_reason=data.get("urgency_reason", ""),
        reasoning=data.get("reasoning", ""),
    )


@app.get("/health")
def health():
    return {"status": "ok", "stt_model": STT_MODEL, "llm_model": LLM_MODEL}


@app.post("/transcribe", response_model=TranscribeResult)
async def transcribe(audio: UploadFile = File(...)):
    """Audio -> raw transcript only (no translation/classification)."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    transcript = _transcribe(audio_bytes, audio.filename)
    return TranscribeResult(transcript=transcript)


@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(request: AnalyzeRequest):
    """Text -> translation + intent classification + urgency (the prompt-engineered core)."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    return _analyze(request.text)


@app.post("/process", response_model=ProcessResult)
async def process(audio: UploadFile = File(...)):
    """Full pipeline: audio -> transcript -> translation + intent + urgency."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    transcript = _transcribe(audio_bytes, audio.filename)
    if not transcript:
        raise HTTPException(status_code=422, detail="Could not transcribe audio")
    analysis = _analyze(transcript)
    return ProcessResult(transcript=transcript, analysis=analysis)


# Serve the frontend from the same app for single-deploy hosting.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
