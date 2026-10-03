"""
============================================
FastAPI Backend
============================================
REST API for the sentiment analysis chatbot.
Endpoints for prediction, batch analysis, and model info.
"""

import logging
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import API_CONFIG, SENTIMENT_LABELS

# ============================================
# Pydantic Models
# ============================================

class TextInput(BaseModel):
    text: str
    model: Optional[str] = None

class BatchInput(BaseModel):
    texts: List[str]
    model: Optional[str] = None

class SentimentResponse(BaseModel):
    status: str
    input: dict
    analysis: dict
    response: dict
    metadata: dict

# ============================================
# Initialize App
# ============================================

app = FastAPI(
    title="Sentiment Analysis Chatbot API",
    description="NLP-powered sentiment analysis with chatbot integration",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
predictor = None
response_generator = None


def initialize_models():
    """Load models and preprocessor on startup."""
    global predictor, response_generator

    from chatbot.predictor import SentimentPredictor
    from chatbot.response_generator import ResponseGenerator

    predictor = SentimentPredictor()
    response_generator = ResponseGenerator()

    # Load preprocessor
    predictor.load_preprocessor()

    # Try loading trained models
    try:
        predictor.load_classical_model("logistic_regression", "tfidf")
        logger.info("Logistic Regression loaded")
    except Exception as e:
        logger.warning(f"Could not load LR: {e}")

    try:
        predictor.load_classical_model("svm", "tfidf")
        logger.info("SVM loaded")
    except Exception as e:
        logger.warning(f"Could not load SVM: {e}")

    try:
        predictor.load_classical_model("naive_bayes", "tfidf")
        logger.info("Naive Bayes loaded")
    except Exception as e:
        logger.warning(f"Could not load NB: {e}")

    try:
        predictor.load_transformer_model()
        logger.info("DistilBERT loaded")
    except Exception as e:
        logger.warning(f"Could not load DistilBERT: {e}")

    available = predictor.get_available_models()
    if available:
        predictor.default_model = available[0]
        logger.info(f"Available models: {available}")
        logger.info(f"Default model: {predictor.default_model}")
    else:
        logger.warning("No models available! Train models first with: python train.py")


@app.on_event("startup")
async def startup_event():
    initialize_models()


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = Path(project_root) / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"message": "Sentiment Analysis Chatbot API", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    available = predictor.get_available_models() if predictor else []
    return {
        "status": "healthy",
        "models_loaded": len(available),
        "available_models": available,
    }


@app.post("/api/predict")
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for a single text input.
    Returns sentiment label, confidence, and suggested response.
    """
    if not predictor or not predictor.get_available_models():
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Train models first with: python train.py",
        )

    # Get sentiment prediction
    result = predictor.predict(input_data.text, input_data.model)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Generate chatbot response
    response = response_generator.generate_response(result)

    return response


@app.post("/api/predict/batch")
async def predict_batch(input_data: BatchInput):
    """Predict sentiment for multiple texts."""
    if not predictor or not predictor.get_available_models():
        raise HTTPException(status_code=503, detail="No models loaded.")

    results = []
    for text in input_data.texts:
        result = predictor.predict(text, input_data.model)
        response = response_generator.generate_response(result)
        results.append(response)

    return {"status": "success", "count": len(results), "results": results}


@app.get("/api/models")
async def list_models():
    """List available models."""
    available = predictor.get_available_models() if predictor else []
    return {"models": available, "default": predictor.default_model if predictor else None}


# Serve static frontend files
frontend_dir = Path(project_root) / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
    )
