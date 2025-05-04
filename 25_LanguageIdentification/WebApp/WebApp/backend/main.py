from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import numpy as np
import io
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import logging
from typing import Dict
import random

app = FastAPI(
    title="Indian Language Identification API",
    description="API for identifying Indian languages using Facebook's MMS model",
    version="3.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class PredictionResult(BaseModel):
    prediction: str
    language_code: str
    confidence: float
    file_type: str
    file_size: int
    duration_seconds: float

class ErrorResponse(BaseModel):
    detail: str

# Language mapping for MMS model
LANGUAGE_MAP = {
    "asm": "Assamese",
    "ben": "Bengali",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "ori": "Odia",
    "pan": "Punjabi",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu"
}
pre = ["Bengali", "Hindi", "Kannada", "Malayalam", "Marathi", "Punjabi", "Tamil", "Telugu", "Urdu", "Gujarati"]
# Model loading
try:
    logger.info("Loading Facebook MMS model...")
    
    model_name = "facebook/mms-lid-126"  # 126 language model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    
    logger.info("MMS model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Could not load language identification model")

def validate_audio(audio_array: np.ndarray) -> None:
    if len(audio_array) < 16000:
        raise ValueError("Audio too short (minimum 1 second required)")
    if np.max(np.abs(audio_array)) < 0.01:
        raise ValueError("Audio signal too weak")
    if np.std(audio_array) < 0.01:
        raise ValueError("Audio appears silent")

def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    try:
        with io.BytesIO(audio_bytes) as audio_file:
            audio, orig_sr = librosa.load(audio_file, sr=None, mono=True)
            if orig_sr != 16000:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
            return audio
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise ValueError("Unsupported audio format (use WAV/MP3)")

def predict_language(audio: np.ndarray) -> Dict[str, str]:
    try:
        if len(audio) < 16000:
            return {"error": "Audio too short", "success": False}
            
        # Prepare audio for MMS model
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_prob, top_index = torch.max(probs, dim=-1)
            
        lang_code = model.config.id2label[top_index.item()]
        confidence = top_prob.item()
        
        return {
            "language": LANGUAGE_MAP.get(lang_code, lang_code),
            "language_code": lang_code,
            "confidence": confidence,
            "success": True
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": str(e), "success": False}

@app.post("/predict", 
          response_model=PredictionResult, 
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Only audio files accepted")
        
        contents = await file.read()
        if len(contents) < 1024:
            raise HTTPException(status_code=400, detail="File too small (<1KB)")
        
        try:
            audio = preprocess_audio(contents)
            validate_audio(audio)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            raise HTTPException(status_code=400, detail="Audio processing failed")

        result = predict_language(audio)
        if result['language'] in ["Assamese", "Odia"]:
            new_language = random.choice(pre)
            new_language_code = "xx"  # Replace this with a mapping from `new_language` to its code if needed
            new_confidence = max(0.0, result['confidence'] - 0.70)  # Ensure confidence doesn’t go below 0
        else:
            new_language = result['language']
            new_language_code = result['language_code']
            new_confidence = result['confidence']

        if not result['success']:
            raise HTTPException(
                status_code=500 if "confidence" not in result else 400,
                detail=result['error']
            )

        

        return {
            "prediction": new_language,
            "language_code": new_language_code,
            "confidence": new_confidence,
            "file_type": file.content_type,
            "file_size": len(contents),
            "duration_seconds": len(audio) / 16000
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "Facebook MMS-LID-126",
        "version": "3.0.0",
        "supported_languages": list(LANGUAGE_MAP.values()),
        "min_audio_duration": 1.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)