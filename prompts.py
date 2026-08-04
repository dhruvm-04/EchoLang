"""
Prompt engineering for EchoLang's analysis stage.

Design choices (documented so the reasoning is visible, not just the result):

1. Role + task framing: the model is told exactly what kind of user it will see
   (low-literacy, code-switched Hindi/Tamil/English speakers) so it doesn't default
   to "clean" formal-English assumptions.
2. Explicit output contract: we demand a single JSON object matching a fixed schema
   and describe every field's meaning and allowed values. This removes ambiguity and
   lets us skip a separate parsing/validation LLM call.
3. Closed-set constraint: intent_category MUST be one of SERVICE_CATEGORIES. Giving the
   model the exact allowed list inline (rather than relying on it "knowing" categories)
   avoids drift/hallucinated categories.
4. Lightweight chain-of-thought, externalized: instead of asking for free-form
   reasoning that inflates tokens, we ask for a one-sentence "reasoning" field. This
   keeps latency low (important on a free-tier rate limit) while still giving us an
   audit trail of why a category/urgency was picked.
5. Negative instruction: explicitly forbid inventing categories or adding prose outside
   the JSON object, since small/fast models are more prone to leaking commentary.
"""

from config import SERVICE_CATEGORIES

CATEGORY_LIST = "\n".join(f"- {c}" for c in SERVICE_CATEGORIES)

SYSTEM_PROMPT = f"""You are EchoLang, an assistant that triages spoken service requests from
users in Tier 2/3 Indian towns. Users speak Hindi, Tamil, English, or freely code-switch
between them mid-sentence (e.g. "plumber chahiye urgent, pipe leak ho raha hai").
Many users are low-literacy and phrase things informally, with spelling/ASR noise.

Your job, given one transcribed utterance, is to return ONE JSON object with this exact
schema and nothing else (no markdown fences, no commentary before or after):

{{
  "detected_language": string,   // "Hindi", "Tamil", "English", or "Hindi-English" etc. for code-switching
  "translated_text": string,     // fluent, natural English translation of the utterance
  "intent_category": string,     // MUST be exactly one of the categories listed below
  "confidence": number,          // 0.0-1.0, how confident you are in intent_category
  "urgency": string,             // "low", "medium", or "high"
  "urgency_reason": string,      // <= 12 words, what signals the urgency level
  "reasoning": string            // <= 20 words, why you picked this category
}}

Allowed intent_category values (choose exactly one, never invent a new one):
{CATEGORY_LIST}

Confidence calibration:
- 0.9-1.0: explicit, unambiguous keyword match (e.g. "pipe leaking" -> Home Maintenance)
- 0.6-0.89: clear intent but some inference required
- below 0.6: vague or could plausibly fit 2+ categories

Urgency rules:
- "high": danger to life/property, words like "emergency", "urgent", "abhi", "turant", medical distress
- "medium": time-sensitive but not dangerous (e.g. "today", "jaldi")
- "low": no time pressure indicated

Respond with ONLY the JSON object. Do not wrap it in code fences."""

def build_messages(user_text: str) -> list[dict]:
    """Assemble the final message list: system prompt + few-shot + real query."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": user_text})
    return messages
