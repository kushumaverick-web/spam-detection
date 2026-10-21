"""
SpamGuard AI — Main FastAPI Server
Provides high-performance REST API endpoints and web interface for email spam detection.
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from detector_engine import detector
import database

# Initialize application
app = FastAPI(
    title="SpamGuard AI API",
    version="1.0.0",
    description="Intelligent Email Spam, Phishing, and Cyber Threat Detection Service."
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Ensure static folder exists
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount Static and Template engines
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Request Models
class EmailScanRequest(BaseModel):
    subject: Optional[str] = Field(default="", description="Email subject line")
    body: Optional[str] = Field(default="", description="Email body text or plain content")
    raw_email: Optional[str] = Field(default="", description="Optional full raw RFC 822 email text")

class BatchEmailItem(BaseModel):
    subject: Optional[str] = ""
    body: Optional[str] = ""
    text: Optional[str] = ""

class BatchScanRequest(BaseModel):
    items: List[BatchEmailItem]

class FeedbackRequest(BaseModel):
    scan_id: Optional[int] = None
    email_text: str
    predicted_verdict: str
    user_corrected_label: str
    notes: Optional[str] = ""


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the interactive web application."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/predict")
async def predict_email(payload: EmailScanRequest):
    """
    Classifies a single email using hybrid AI (TF-IDF Ensemble + Heuristic Threat Signatures).
    """
    subject = payload.subject or ""
    body = payload.body or ""
    raw = payload.raw_email or ""

    if not (subject.strip() or body.strip() or raw.strip()):
        raise HTTPException(status_code=400, detail="Subject, body, or raw email content is required.")

    result = detector.predict(subject=subject, body=body, raw_email=raw)

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))

    # Save record to database
    preview_snippet = (subject + " " + body) if (subject or body) else raw
    scan_id = database.save_scan(
        subject=subject or raw[:50],
        preview=preview_snippet[:200],
        verdict=result["verdict"],
        verdict_badge=result["verdict_badge"],
        spam_probability=result["spam_probability"],
        risk_level=result["risk_level"],
        url_count=result["heuristics"]["url_count"],
        threat_count=len(result["heuristics"]["threat_flags"])
    )
    result["scan_id"] = scan_id

    return JSONResponse(content=result)

@app.post("/api/batch-scan")
async def batch_scan_emails(payload: BatchScanRequest):
    """
    Classifies a batch of up to 500 emails in a single request.
    """
    if not payload.items:
        raise HTTPException(status_code=400, detail="Batch items list cannot be empty.")

    results = []
    spam_count = 0
    ham_count = 0

    for idx, item in enumerate(payload.items[:500]):
        text = item.text or f"{item.subject} {item.body}"
        res = detector.predict(subject=item.subject or "", body=item.body or item.text or "")
        
        is_spam = res["verdict"] == "SPAM" or res["spam_probability"] >= 50.0
        if is_spam:
            spam_count += 1
        else:
            ham_count += 1

        results.append({
            "index": idx + 1,
            "preview": text[:80] + ("..." if len(text) > 80 else ""),
            "verdict": res["verdict"],
            "spam_probability": res["spam_probability"],
            "risk_level": res["risk_level"],
            "threat_count": len(res["heuristics"]["threat_flags"])
        })

    return {
        "status": "success",
        "total": len(results),
        "spam_count": spam_count,
        "ham_count": ham_count,
        "results": results
    }

@app.get("/api/metrics")
async def get_model_metrics():
    """
    Returns validation metrics, confusion matrix, and feature importances.
    """
    return JSONResponse(content=detector.metrics)

@app.get("/api/history")
async def get_scan_history(limit: int = 15):
    """
    Returns recent scans from local SQLite database.
    """
    scans = database.get_recent_scans(limit=min(limit, 50))
    return JSONResponse(content={"scans": scans})

@app.get("/api/stats")
async def get_app_stats():
    """
    Returns overall platform usage metrics.
    """
    stats = database.get_total_stats()
    return JSONResponse(content=stats)

@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackRequest):
    """
    Saves user feedback for misclassified emails to aid active learning retraining.
    """
    fb_id = database.save_feedback(
        scan_id=payload.scan_id,
        email_text=payload.email_text,
        predicted_verdict=payload.predicted_verdict,
        user_corrected_label=payload.user_corrected_label,
        notes=payload.notes or ""
    )
    return JSONResponse(content={"status": "success", "feedback_id": fb_id})

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for container orchestrators and uptime monitors.
    """
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "service": "SpamGuard AI",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
