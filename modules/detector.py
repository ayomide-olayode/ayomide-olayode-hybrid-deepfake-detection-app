# modules/deepfake_detector.py
import logging
from .video_processor import VideoProcessor
from .text_processor import TextProcessor
from .audio_processor import AudioProcessor
from .fusion import HybridFusion

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Multimodal deepfake detection system"""
    
    def __init__(self):
        logger.info("Initializing deepfake detection system")
        self.video_processor = VideoProcessor()
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        self.fusion_layer = HybridFusion()
    
    def extract_modality_features(self, video_path: str) -> dict:
        """Extract features from all three modalities"""
        results = {}
        
        try:
            # Process video features
            frames = self.video_processor.extract_frames(video_path)
            video_results = self.video_processor.process_frames(frames)
            results["video"] = video_results
            
            # Process text from frames and audio transcription
            audio_transcription = self.audio_processor.transcribe_audio(video_path)
            text_results = self.text_processor.process(frames, audio_transcription)
            results["text"] = text_results
            
            # Process audio features
            audio_results = self.audio_processor.process(video_path)
            results["audio"] = audio_results
            
            return results
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return results
    
    def detect_deepfake(self, video_path: str) -> dict:
        """Main pipeline for deepfake detection"""
        try:
            # Step 1: Feature extraction
            modality_features = self.extract_modality_features(video_path)
            
            # Step 2: Cross-modal fusion
            fusion_results = self.fusion_layer.fuse(
                modality_features["video"],
                modality_features["text"],
                modality_features["audio"]
            )
            
            # Step 3: Compile results
            return {
                "deepfake_probability": fusion_results['fused_confidence'],
                "modality_results": modality_features,
                "analysis_times": {
                    "video": modality_features["video"].get("processing_time", 0),
                    "audio": modality_features["audio"].get("processing_time", 0),
                    "text": modality_features["text"].get("processing_time", 0)
                },
                "fusion_method": fusion_results.get('method', 'Advanced Fusion')
            }
            
        except Exception as e:
            logger.error(f"Detection pipeline failed: {e}")
            return {
                "error": str(e),
                "deepfake_probability": 0.5
            }