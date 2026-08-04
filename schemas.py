from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw (possibly code-switched) user text, Hindi/Tamil/English")


class AnalyzeResult(BaseModel):
    original_text: str
    detected_language: str
    translated_text: str
    intent_category: str
    confidence: float
    urgency: str
    urgency_reason: str
    reasoning: str


class TranscribeResult(BaseModel):
    transcript: str
    language: Optional[str] = None


class ProcessResult(BaseModel):
    transcript: str
    analysis: AnalyzeResult
