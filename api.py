import os
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.ensemble_model import load_or_train, predict
from backend.groq_verifier import configure_groq, groq_fact_check

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Credible 2.0 API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
ML_CONFIDENCE_THRESHOLD = 0.80
PAYLOAD = None
GROQ_CONFIGURED = False

@app.on_event("startup")
def startup_event():
    global PAYLOAD, GROQ_CONFIGURED
    print("Loading ML Ensemble Model...")
    PAYLOAD = load_or_train()
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key and api_key != "your_groq_api_key_here":
        try:
            configure_groq(api_key)
            GROQ_CONFIGURED = True
            print("Groq API configured successfully.")
        except Exception as e:
            print(f"Failed to configure Groq: {e}")

class AnalyzeRequest(BaseModel):
    text: str
    use_groq: bool = True

@app.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    word_count = len(req.text.split())
    
    # 1. Always run ML Model first
    ml_result = predict(req.text, PAYLOAD)
    
    # 2. Check if we need to escalate to Groq
    escalate = (
        req.use_groq and 
        GROQ_CONFIGURED and 
        (ml_result["confidence"] < ML_CONFIDENCE_THRESHOLD or word_count < 20)
    )
    
    if escalate:
        # Run Groq Fact Check
        groq_result = groq_fact_check(req.text)
        if groq_result is not None:
            return {
                "engine": f"Groq ({os.getenv('GROQ_MODEL', 'qwen/qwen3.8-27b')})",
                "label": groq_result["label"],
                "confidence": groq_result["confidence"],
                "reasoning": groq_result.get("reasoning", "No reasoning provided by Groq."),
            }
        else:
            # Fallback if Groq throws an API error
            return {
                "engine": "ML Ensemble (LLM Unavailable)",
                "label": ml_result["label"],
                "confidence": ml_result["confidence"],
                "reasoning": f"LLM escalation unavailable. Based on ML consensus. (Fake Prob: {ml_result['fake_prob']:.0%}, Real Prob: {ml_result['real_prob']:.0%})",
            }
    else:
        # Return ML Results
        return {
            "engine": "ML Ensemble",
            "label": ml_result["label"],
            "confidence": ml_result["confidence"],
            "reasoning": f"Based on machine learning consensus across 5 models. (Fake Probability: {ml_result['fake_prob']:.0%}, Real Probability: {ml_result['real_prob']:.0%})",
        }

@app.post("/analyze-batch")
async def analyze_batch(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Invalid CSV file format")
        
    if "text" not in df.columns:
        raise HTTPException(400, "CSV must contain a 'text' column")
        
    results = []
    for text in df["text"].dropna().astype(str):
        r = predict(text, PAYLOAD)
        results.append({
            "text": text[:80] + "...",
            "label": r["label"],
            "confidence": f"{r['confidence']:.1%}"
        })
        
    return {"results": results}

# Serve the Frontend directly from FastAPI to keep it simple!
@app.get("/")
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
