"""
gemini_verifier.py  -  Hybrid Fact Checker using Groq API
==========================================================
Replaces Gemini with Groq (free, fast, no quota issues).
Uses Llama 3.3 70B model for fact checking.

Get your FREE API key at: https://console.groq.com
"""

import json
from groq import Groq

# ── Global client ─────────────────────────────────────────────────────────────
_client = None


# ── Configure Groq ────────────────────────────────────────────────────────────
def configure_groq(api_key: str):
    """Call this once at app startup with your Groq API key.
    Named configure_gemini to keep app.py imports unchanged.
    """
    global _client
    _client = Groq(api_key=api_key)


# ── Fact Check using Groq ─────────────────────────────────────────────────────
def groq_fact_check(text: str) -> dict:
    """
    Send text to Groq (Llama 3.3 70B) for fact checking.

    Returns
    -------
    {
        "label"      : "Fake" | "Real" | "Unverifiable",
        "confidence" : float (0.0 - 1.0),
        "reasoning"  : str,
        "red_flags"  : list[str],
        "sources"    : list[str],
        "engine"     : "Groq (Llama 3.3 70B)"
    }
    """
    global _client

    if _client is None:
        return {
            "label":      "Unverifiable",
            "confidence": 0.5,
            "reasoning":  "Groq API not configured. Please enter your API key in the sidebar.",
            "red_flags":  [],
            "sources":    [],
            "engine":     "Groq (Llama 3.3 70B)",
        }

    try:
        prompt = f"""You are a professional fact-checker and misinformation detection expert.

Analyze the following news text or claim and determine if it is GENUINE (real/true) or FAKE (false/misleading/misinformation).

TEXT TO ANALYZE:
\"\"\"{text}\"\"\"

Instructions:
1. Use your knowledge to fact-check this claim
2. Look for signs of misinformation:
- Sensational or alarmist language
- Unverifiable anonymous sources
- Claims that contradict known facts
- Conspiracy theory patterns
- Exaggerated statistics or numbers
3. Make a final verdict

Respond ONLY in this exact JSON format with no extra text or markdown:
{{
    "verdict": "Fake" or "Real" or "Unverifiable",
    "confidence": <number between 0.0 and 1.0>,
    "reasoning": "<clear explanation in 2-3 sentences why this is fake or real>",
    "red_flags": ["<flag1>", "<flag2>"],
    "sources": ["<source1>", "<source2>"]
}}"""

        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert fact-checker. Always respond with valid JSON only. No markdown, no extra text.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,    # low temperature = more consistent outputs
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()

        # Clean up response in case model adds markdown
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)

        return {
            "label":      data.get("verdict", "Unverifiable"),
            "confidence": float(data.get("confidence", 0.5)),
            "reasoning":  data.get("reasoning", "No reasoning provided."),
            "red_flags":  data.get("red_flags", []),
            "sources":    data.get("sources", []),
            "engine":     "Groq (Llama 3.3 70B)",
        }

    except json.JSONDecodeError:
        raw_text = response.choices[0].message.content if 'response' in dir() else ""
        return _parse_text_response(raw_text)

    except Exception as e:
        return {
            "label":      "Unverifiable",
            "confidence": 0.5,
            "reasoning":  f"Groq API error: {str(e)}",
            "red_flags":  [],
            "sources":    [],
            "engine":     "Groq (Llama 3.3 70B)",
        }


# ── Fallback text parser ──────────────────────────────────────────────────────
def _parse_text_response(text: str) -> dict:
    """Fallback parser if model returns plain text instead of JSON."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["fake", "false", "misleading", "misinformation", "fabricated"]):
        label      = "Fake"
        confidence = 0.75
    elif any(w in text_lower for w in ["real", "true", "genuine", "accurate", "correct", "verified"]):
        label      = "Real"
        confidence = 0.75
    else:
        label      = "Unverifiable"
        confidence = 0.5

    return {
        "label":      label,
        "confidence": confidence,
        "reasoning":  text[:300] if text else "Could not parse response.",
        "red_flags":  [],
        "sources":    [],
        "engine":     "Groq (Llama 3.3 70B)",
    }


# ── Test function ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = input("Enter your Groq API key: ").strip()
    configure_groq(api_key)

    tests = [
        "Narendra Modi is the Prime Minister of India",
        "Government is secretly putting microchips in COVID vaccines",
        "NASA successfully launched the James Webb Space Telescope in 2021",
        "Drinking bleach cures cancer according to doctors",
    ]

    for test in tests:
        print(f"\nText: {test[:60]}...")
        result = groq_fact_check(test)
        print(f"  Verdict    : {result['label']}")
        print(f"  Confidence : {result['confidence']:.0%}")
        print(f"  Reasoning  : {result['reasoning'][:100]}...")