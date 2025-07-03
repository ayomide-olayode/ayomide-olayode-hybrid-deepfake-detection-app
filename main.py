from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import time
import cv2
import logging
import json
import traceback
from modules.detector import DeepfakeDetector  # Import the detector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deepfake Detection API",
    description="Multi-modal deepfake detection system",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

@app.get("/")
async def root():
    return {
        "message": "Deepfake Detection API v3.0", 
        "features": ["Multi-modal Analysis", "Cross-Modal Fusion", "Explainable AI"],
        "models": ["EfficientNet-B4", "Whisper", "BERT", "CLIP", "EasyOCR"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/models")
async def get_available_models():
    return {
        "video": {
            "name": "EfficientNet-B4",
            "description": "Frame analysis for visual artifacts",
            "input": "Video frames",
            "output": "Visual confidence score"
        },
        "audio": {
            "name": "Whisper + Audio Analysis",
            "description": "Speech analysis and audio artifacts",
            "input": "Audio stream",
            "output": "Audio confidence score"
        },
        "text": {
            "name": "EasyOCR + BERT + CLIP",
            "description": "Text extraction and semantic consistency",
            "input": "Video frames + transcription",
            "output": "Text confidence score"
        },
        "multimodal": {
            "name": "Hybrid Fusion",
            "description": "Cross-modal attention + GNN fusion",
            "input": "All modalities",
            "output": "Fused confidence score"
        }
    }

@app.post("/analyze/video")
async def analyze_video_only(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """Analyze video using EfficientNet-B4"""
    return await analyze_with_model(file, "video", confidence_threshold)

@app.post("/analyze/audio")
async def analyze_audio_only(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """Analyze audio using Whisper"""
    return await analyze_with_model(file, "audio", confidence_threshold)

@app.post("/analyze/text")
async def analyze_text_only(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """Analyze text using EasyOCR + BERT + CLIP"""
    return await analyze_with_model(file, "text", confidence_threshold)

async def analyze_with_model(file: UploadFile, model_type: str, confidence_threshold: float):
    """Analyze uploaded video with specified model"""
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size (100MB limit)
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Check video duration
        duration = get_video_duration(tmp_file_path)
        if duration > 30:
            raise HTTPException(
                status_code=400, 
                detail=f"Video duration ({duration:.1f}s) exceeds 30 second limit"
            )
        
        # Import and initialize the appropriate processor
        start_time = time.time()
        
        if model_type == "video":
            from modules.video_processor import VideoProcessor
            processor = VideoProcessor()
        elif model_type == "audio":
            from modules.audio_processor import AudioProcessor
            processor = AudioProcessor()
        elif model_type == "text":
            from modules.text_processor import TextProcessor
            processor = TextProcessor()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # Run analysis
        logger.info(f"Running {model_type} analysis on {file.filename}")
        results = processor.process(tmp_file_path)
        analysis_time = time.time() - start_time
        
        # Determine if deepfake
        confidence = results.get('confidence', 0.5)
        is_deepfake = confidence > confidence_threshold
        
        return {
            "filename": file.filename,
            "model_type": model_type,
            "is_deepfake": is_deepfake,
            "confidence": confidence,
            "threshold_used": confidence_threshold,
            "analysis_time_seconds": analysis_time,
            "video_duration_seconds": duration,
            "results": results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/analyze/multimodal")
async def analyze_multimodal(
    file: UploadFile = File(...),
    enable_video: bool = Form(True),
    enable_audio: bool = Form(True),
    enable_text: bool = Form(True),
    confidence_threshold: float = Form(0.5),
    fusion_method: str = Form("Advanced Fusion")
):
    """Multi-modal deepfake analysis endpoint"""
    if not any([enable_video, enable_audio, enable_text]):
        raise HTTPException(status_code=400, detail="At least one modality must be enabled")
    
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size (100MB limit)
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Check video duration
        duration = get_video_duration(tmp_file_path)
        if duration > 30:
            raise HTTPException(
                status_code=400, 
                detail=f"Video duration ({duration:.1f}s) exceeds 30 second limit"
            )
        
        # Initialize detector
        detector = DeepfakeDetector()
        
        # Run multimodal detection
        start_time = time.time()
        result = detector.detect_deepfake(tmp_file_path)
        analysis_time = time.time() - start_time
        
        # Extract confidence
        confidence = result.get('deepfake_probability', 0.5)
        is_deepfake = confidence > confidence_threshold
        
        return {
            "filename": file.filename,
            "is_deepfake": is_deepfake,
            "confidence": confidence,
            "threshold_used": confidence_threshold,
            "fusion_method": fusion_method,
            "analysis_time_seconds": analysis_time,
            "video_duration_seconds": duration,
            "results": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Multi-modal analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)