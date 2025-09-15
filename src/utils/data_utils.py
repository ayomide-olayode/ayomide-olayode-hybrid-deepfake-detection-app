"""
Data utility functions for the HybridDeepfakeDetector project.
Handles video processing, face extraction, and audio transcription.
"""

import cv2
import os
import numpy as np
from pathlib import Path
import whisper
from PIL import Image
import json
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video processing operations including frame extraction and face detection."""
    
    def __init__(self, cascade_path: str = None):
        """
        Initialize VideoProcessor with face cascade classifier.
        
        Args:
            cascade_path: Path to OpenCV cascade file. If None, uses default.
        """
        # TODO: Update cascade path if you have a custom location
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Could not load face cascade from {cascade_path}")
        
        logger.info(f"Loaded face cascade from {cascade_path}")
    
    def extract_frames(self, video_path: str, fps: int = 1) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video_path: Path to input video file
            fps: Frames per second to extract (default: 1)
            
        Returns:
            List of extracted frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps) if video_fps > fps else 1
        
        frame_count = 0
        extracted_count = 0
        
        logger.info(f"Extracting frames from {video_path} at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                frames.append(frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {extracted_count} frames from {frame_count} total frames")
        
        return frames
    
    def detect_and_crop_faces(self, frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """
        Detect faces in frame and crop them to target size.
        
        Args:
            frame: Input frame as numpy array
            target_size: Target size for cropped faces (width, height)
            
        Returns:
            List of cropped face images
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        cropped_faces = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face = frame[y:y+h, x:x+w]
            
            # Resize to target size
            face_resized = cv2.resize(face, target_size)
            cropped_faces.append(face_resized)
        
        return cropped_faces
    
    def process_video_frames(self, video_path: str, output_dir: str, fps: int = 1, 
                           target_size: Tuple[int, int] = (224, 224)) -> int:
        """
        Process video: extract frames and crop faces.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save processed frames
            fps: Frames per second to extract
            target_size: Target size for face crops
            
        Returns:
            Number of face images saved
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        frames = self.extract_frames(video_path, fps)
        
        saved_count = 0
        
        for frame_idx, frame in enumerate(frames):
            # Detect and crop faces
            faces = self.detect_and_crop_faces(frame, target_size)
            
            # Save each detected face
            for face_idx, face in enumerate(faces):
                filename = f"frame_{frame_idx:04d}_face_{face_idx:02d}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Convert BGR to RGB for saving
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                pil_image.save(filepath, quality=95)
                
                saved_count += 1
        
        logger.info(f"Saved {saved_count} face images to {output_dir}")
        return saved_count

class AudioProcessor:
    """Handles audio processing and transcription using Whisper."""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize AudioProcessor with Whisper model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        # TODO: Change model size based on your accuracy/speed requirements
        self.model_name = model_name
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        logger.info("Whisper model loaded successfully")
    
    def transcribe_video(self, video_path: str) -> dict:
        """
        Transcribe audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing transcription results
        """
        logger.info(f"Transcribing audio from {video_path}")
        
        try:
            result = self.model.transcribe(video_path)
            
            # Extract relevant information
            transcription_data = {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": []
            }
            
            # Add segment information if available
            if "segments" in result:
                for segment in result["segments"]:
                    transcription_data["segments"].append({
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "text": segment.get("text", "").strip()
                    })
            
            logger.info(f"Transcription completed. Text length: {len(transcription_data['text'])} characters")
            return transcription_data
            
        except Exception as e:
            logger.error(f"Error transcribing {video_path}: {str(e)}")
            return {
                "text": "",
                "language": "unknown",
                "segments": [],
                "error": str(e)
            }
    
    def save_transcription(self, transcription_data: dict, output_path: str):
        """
        Save transcription data to file.
        
        Args:
            transcription_data: Transcription results dictionary
            output_path: Path to save transcription
        """
        # Save as JSON for structured data
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)
        
        # Save plain text for easy reading
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription_data["text"])
        
        logger.info(f"Transcription saved to {output_path} and {json_path}")

def is_video_processed(video_path: str, base_output_dir: str) -> bool:
    """
    Check if a video has already been processed.
    
    Args:
        video_path: Path to video file
        base_output_dir: Base output directory
        
    Returns:
        True if video is already processed, False otherwise
    """
    video_name = Path(video_path).stem
    frames_dir = os.path.join(base_output_dir, f"{video_name}_frames")
    transcript_path = os.path.join(base_output_dir, f"{video_name}_transcript.txt")
    
    # Check if both frames directory and transcript exist
    frames_exist = os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0
    transcript_exists = os.path.exists(transcript_path)
    
    return frames_exist and transcript_exists

def get_video_files(directory: str, extensions: List[str]) -> List[str]:
    """
    Get all video files from directory with specified extensions.
    
    Args:
        directory: Directory to search
        extensions: List of video file extensions
        
    Returns:
        List of video file paths
    """
    video_files = []
    
    for ext in extensions:
        pattern = f"*{ext}"
        video_files.extend(Path(directory).glob(pattern))
        # Also check uppercase extensions
        pattern = f"*{ext.upper()}"
        video_files.extend(Path(directory).glob(pattern))
    
    return [str(f) for f in video_files]

def create_processing_summary(processed_videos: List[dict], output_path: str):
    """
    Create a summary of processed videos.
    
    Args:
        processed_videos: List of processing results
        output_path: Path to save summary
    """
    summary = {
        "total_videos": len(processed_videos),
        "successful": sum(1 for v in processed_videos if v.get("success", False)),
        "failed": sum(1 for v in processed_videos if not v.get("success", False)),
        "total_faces": sum(v.get("face_count", 0) for v in processed_videos),
        "videos": processed_videos
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processing summary saved to {output_path}")
    logger.info(f"Total: {summary['total_videos']}, Success: {summary['successful']}, Failed: {summary['failed']}")
