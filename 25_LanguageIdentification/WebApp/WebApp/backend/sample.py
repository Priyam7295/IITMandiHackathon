from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import numpy as np
import io
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import logging
from typing import Dict
import os
from torch import nn

app = FastAPI(
    title="Indian Language Identification API",
    description="API for identifying Indian languages using custom Hubert model",
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

# Language mapping
LABEL_MAP = {
    0: {"language": "Kannada", "code": "kan"},
    1: {"language": "Marathi", "code": "mar"},
    2: {"language": "Punjabi", "code": "pan"},
    3: {"language": "Telugu", "code": "tel"},
    4: {"language": "Gujarati", "code": "guj"},
    5: {"language": "Malayalam", "code": "mal"},
    6: {"language": "Urdu", "code": "urd"},
    7: {"language": "Tamil", "code": "tam"},
    8: {"language": "Hindi", "code": "hin"},
    9: {"language": "Bengali", "code": "ben"}
}

# Custom Hubert Model Class
class HubertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(HubertClassifier, self).__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.classifier = nn.Sequential(
            nn.Linear(self.hubert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values):
        outputs = self.hubert(input_values).last_hidden_state
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled)

# Model initialization
try:
    logger.info("Loading custom Hubert model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(LABEL_MAP)
    
    # Initialize model
    model = HubertClassifier(num_labels=num_classes)
    
    # Load pretrained weights (update path as needed)
    model_path = "hubert.pt"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    logger.info("Hubert model loaded successfully")
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

def predict_language(audio: np.ndarray, chunk_duration_sec: int = 5) -> Dict[str, str]:
    try:
        if len(audio) < 16000:
            return {"error": "Audio too short", "success": False}
            
        sample_rate = 16000
        chunk_samples = chunk_duration_sec * sample_rate
        total_len = len(audio)
        num_chunks = (total_len + chunk_samples - 1) // chunk_samples
        
        # Convert numpy array to torch tensor
        waveform = torch.from_numpy(audio).float()
        
        # Process each chunk
        predictions = []
        confidences = []
        probsA=[]
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, total_len)
            chunk = waveform[start:end]
            
            if chunk.shape[0] < chunk_samples:
                pad_len = chunk_samples - chunk.shape[0]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))
            
            inputs = feature_extractor(
                chunk.numpy(), 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_values.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                top_prob, top_index = torch.max(probs, dim=-1)
                probsA.append(probs)
                predictions.append(top_index.item())
                confidences.append(top_prob.item())
        
        # Aggregate results from all chunks
        if not predictions:
            return {"error": "No valid predictions", "success": False}
            
        # Get most frequent prediction
        final_pred = max(set(predictions), key=predictions.count)
        avg_confidence = sum(confidences) / len(confidences)
        
        language_info = LABEL_MAP.get(final_pred, {"language": "Unknown", "code": "unk"})
        
        return {
            "language": language_info["language"],
            "language_code": language_info["code"],
            "confidence": avg_confidence,
            "success": True,
            "probs": probsA
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
        
        if not result['success']:
            raise HTTPException(
                status_code=500 if "confidence" not in result else 400,
                detail=result['error']
            )

        return {
            "prediction": result['language'],
            "language_code": result['language_code'],
            "confidence": result['confidence'],
            "file_type": file.content_type,
            "file_size": len(contents),
            "duration_seconds": len(audio) / 16000,
            "probs": result['probs']
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
        "model": "Custom Hubert Classifier",
        "version": "3.0.0",
        "supported_languages": [info["language"] for info in LABEL_MAP.values()],
        "min_audio_duration": 1.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090, reload=True)