"""
Streamlit web application for HybridDeepfakeDetector.
Provides an interactive interface for uploading videos and getting deepfake predictions.
"""

import streamlit as st
import os
import sys
import tempfile
import time
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional

import torch
import numpy as np
from PIL import Image
import cv2

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import config
from src.models.hybrid_model import create_model, get_model_config
from src.utils.data_utils import VideoProcessor, AudioProcessor
from src.utils.model_utils import ModelCheckpoint
from src.datasets.deepfake_dataset import DeepfakeDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Hybrid Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .real-prediction {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    
    .fake-prediction {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .processing-step {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
    }
    
    .step-complete {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    
    .step-error {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class DeepfakeDetectorApp:
    """Main application class for the Streamlit interface."""
    
    def __init__(self):
        """Initialize the application."""
        self.model = None
        self.device = None
        self.video_processor = None
        self.audio_processor = None
        self.model_loaded = False
        
        # Initialize session state
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
        if 'transcript_text' not in st.session_state:
            st.session_state.transcript_text = ""
    
    def load_model(self) -> bool:
        """
        Load the trained model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.model_loaded:
            return True
        
        try:
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create model
            model_config = get_model_config()
            self.model = create_model(model_config)
            
            # Load trained weights
            model_path = config.MODEL_OUTPUT_DIR / "hybrid_deepfake_detector_best.pth"
            
            if not model_path.exists():
                st.error(f"Trained model not found at {model_path}")
                st.error("Please train the model first using the training script.")
                return False
            
            # Load checkpoint
            checkpoint_manager = ModelCheckpoint(str(config.MODEL_OUTPUT_DIR))
            checkpoint = checkpoint_manager.load_checkpoint(
                model=self.model,
                checkpoint_path=str(model_path)
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize processors
            self.video_processor = VideoProcessor()
            self.audio_processor = AudioProcessor(model_name=config.WHISPER_MODEL)
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            logger.error(f"Error loading model: {e}")
            return False
    
    def process_video(self, video_file, progress_bar, status_text) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """
        Process uploaded video file.
        
        Args:
            video_file: Uploaded video file
            progress_bar: Streamlit progress bar
            status_text: Streamlit status text element
            
        Returns:
            Tuple of (processed_frames, transcript_text)
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.read())
                temp_video_path = tmp_file.name
            
            # Update progress
            progress_bar.progress(10)
            status_text.markdown('<div class="processing-step">📹 Extracting frames from video...</div>', 
                               unsafe_allow_html=True)
            
            # Extract frames
            frames = self.video_processor.extract_frames(
                temp_video_path, 
                fps=config.FRAMES_PER_SECOND
            )
            
            if not frames:
                st.error("No frames could be extracted from the video.")
                return None, None
            
            progress_bar.progress(30)
            status_text.markdown('<div class="processing-step">👤 Detecting and cropping faces...</div>', 
                               unsafe_allow_html=True)
            
            # Process frames to extract faces
            processed_frames = []
            for frame in frames:
                faces = self.video_processor.detect_and_crop_faces(
                    frame, 
                    target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
                )
                if faces:
                    # Use the first detected face
                    processed_frames.append(faces[0])
            
            if not processed_frames:
                st.error("No faces detected in the video.")
                return None, None
            
            progress_bar.progress(50)
            status_text.markdown('<div class="processing-step">🎵 Transcribing audio...</div>', 
                               unsafe_allow_html=True)
            
            # Transcribe audio
            transcription_data = self.audio_processor.transcribe_video(temp_video_path)
            transcript_text = transcription_data.get('text', '')
            
            progress_bar.progress(70)
            status_text.markdown('<div class="processing-step">🔄 Preparing data for model...</div>', 
                               unsafe_allow_html=True)
            
            # Convert frames to tensor
            frame_tensors = []
            for frame in processed_frames[:config.MAX_FRAMES if hasattr(config, 'MAX_FRAMES') else 10]:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                # Apply transforms (same as dataset)
                transform = self._get_transform()
                frame_tensor = transform(pil_image)
                frame_tensors.append(frame_tensor)
            
            # Pad or truncate to fixed number of frames
            max_frames = 10
            while len(frame_tensors) < max_frames:
                # Pad with black frames
                black_frame = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                frame_tensors.append(black_frame)
            
            frame_tensors = frame_tensors[:max_frames]
            
            # Stack frames into tensor
            frames_tensor = torch.stack(frame_tensors).unsqueeze(0)  # Add batch dimension
            
            progress_bar.progress(90)
            
            # Clean up temporary file
            os.unlink(temp_video_path)
            
            return frames_tensor, transcript_text
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Error processing video: {e}")
            return None, None
    
    def _get_transform(self):
        """Get image transform for preprocessing."""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, frames_tensor: torch.Tensor, transcript_text: str) -> Dict:
        """
        Make prediction using the loaded model.
        
        Args:
            frames_tensor: Processed video frames
            transcript_text: Transcribed text
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Tokenize text
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Handle empty text
            if not transcript_text.strip():
                transcript_text = "[EMPTY]"
            
            # Tokenize
            encoding = tokenizer(
                transcript_text,
                truncation=True,
                padding='max_length',
                max_length=config.MAX_TEXT_LENGTH,
                return_tensors='pt'
            )
            
            text_data = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            }
            
            # Move to device
            frames_tensor = frames_tensor.to(self.device)
            text_data = {k: v.to(self.device) for k, v in text_data.items()}
            
            # Make prediction
            with torch.no_grad():
                probability = self.model(frames_tensor, text_data)
                probability = probability.squeeze().cpu().item()
            
            # Convert to prediction
            prediction = "FAKE" if probability >= 0.5 else "REAL"
            confidence = probability if probability >= 0.5 else (1 - probability)
            
            return {
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'is_fake': probability >= 0.5
            }
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            logger.error(f"Error making prediction: {e}")
            return None
    
    def display_prediction(self, result: Dict):
        """
        Display prediction results.
        
        Args:
            result: Prediction results dictionary
        """
        prediction = result['prediction']
        confidence = result['confidence']
        probability = result['probability']
        
        # Main prediction display
        if result['is_fake']:
            st.markdown(f'''
            <div class="prediction-box fake-prediction">
                🚨 DEEPFAKE DETECTED 🚨<br>
                Confidence: {confidence:.1%}
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="prediction-box real-prediction">
                ✅ AUTHENTIC VIDEO ✅<br>
                Confidence: {confidence:.1%}
            </div>
            ''', unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", prediction)
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.metric("Raw Score", f"{probability:.3f}")
        
        # Confidence bar
        st.subheader("Confidence Visualization")
        
        # Create confidence bar
        if result['is_fake']:
            color = "#dc3545"  # Red for fake
            bar_value = probability
        else:
            color = "#28a745"  # Green for real
            bar_value = 1 - probability
        
        st.markdown(f'''
        <div style="background-color: #e9ecef; border-radius: 10px; padding: 5px;">
            <div style="background-color: {color}; width: {bar_value*100}%; height: 20px; 
                        border-radius: 10px; transition: width 0.3s ease;"></div>
        </div>
        <p style="text-align: center; margin-top: 5px;">
            Real ← {probability:.3f} → Fake
        </p>
        ''', unsafe_allow_html=True)
    
    def run(self):
        """Run the main Streamlit application."""
        # Header
        st.markdown('<h1 class="main-header">🔍 Hybrid Deepfake Detector</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; color: #666;">
            Upload a video to detect if it contains deepfake content using advanced AI analysis
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("ℹ️ About")
            st.markdown("""
            This application uses a hybrid deep learning model that analyzes both:
            
            **🎥 Visual Features**
            - Face detection and analysis
            - Frame-by-frame examination
            - EfficientNet-B0 backbone
            
            **📝 Audio Features**
            - Speech transcription
            - Text analysis with DistilBERT
            - Lip-sync detection
            
            **🔬 Model Details**
            - Combines visual and textual modalities
            - Trained on deepfake datasets
            - Real-time processing capability
            """)
            
            st.header("📋 Instructions")
            st.markdown("""
            1. Upload a video file (MP4, AVI, MOV, etc.)
            2. Wait for processing to complete
            3. View the prediction results
            4. Check the transcribed text
            """)
            
            # Model status
            st.header("🤖 Model Status")
            if self.load_model():
                st.success("✅ Model loaded successfully")
                st.info(f"Device: {self.device}")
            else:
                st.error("❌ Model not available")
                return
        
        # Main content
        # File uploader
        st.subheader("📁 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Upload a video file to analyze for deepfake content"
        )
        
        if uploaded_file is not None:
            # Display video info
            st.subheader("📹 Uploaded Video")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display video
                st.video(uploaded_file)
            
            with col2:
                # Video details
                st.write("**File Details:**")
                st.write(f"Name: {uploaded_file.name}")
                st.write(f"Size: {uploaded_file.size / (1024*1024):.1f} MB")
                st.write(f"Type: {uploaded_file.type}")
            
            # Process button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                # Reset session state
                st.session_state.processing_complete = False
                st.session_state.prediction_result = None
                st.session_state.transcript_text = ""
                
                # Processing section
                st.subheader("⚙️ Processing Video")
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video
                frames_tensor, transcript_text = self.process_video(
                    uploaded_file, progress_bar, status_text
                )
                
                if frames_tensor is not None and transcript_text is not None:
                    # Make prediction
                    status_text.markdown('<div class="processing-step">🧠 Running AI analysis...</div>', 
                                       unsafe_allow_html=True)
                    progress_bar.progress(95)
                    
                    result = self.predict(frames_tensor, transcript_text)
                    
                    if result:
                        progress_bar.progress(100)
                        status_text.markdown('<div class="processing-step step-complete">✅ Analysis complete!</div>', 
                                           unsafe_allow_html=True)
                        
                        # Store results in session state
                        st.session_state.processing_complete = True
                        st.session_state.prediction_result = result
                        st.session_state.transcript_text = transcript_text
                        
                        # Small delay for better UX
                        time.sleep(1)
                        st.rerun()
                    else:
                        status_text.markdown('<div class="processing-step step-error">❌ Analysis failed</div>', 
                                           unsafe_allow_html=True)
                else:
                    status_text.markdown('<div class="processing-step step-error">❌ Video processing failed</div>', 
                                       unsafe_allow_html=True)
        
        # Display results if processing is complete
        if st.session_state.processing_complete and st.session_state.prediction_result:
            st.subheader("🎯 Analysis Results")
            
            # Display prediction
            self.display_prediction(st.session_state.prediction_result)
            
            # Display transcript
            if st.session_state.transcript_text:
                st.subheader("📝 Transcribed Audio")
                
                with st.expander("View Transcript", expanded=False):
                    st.text_area(
                        "Transcribed Text",
                        value=st.session_state.transcript_text,
                        height=150,
                        disabled=True
                    )
                    
                    # Transcript stats
                    word_count = len(st.session_state.transcript_text.split())
                    char_count = len(st.session_state.transcript_text)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Word Count", word_count)
                    with col2:
                        st.metric("Character Count", char_count)
            else:
                st.info("No audio transcript available for this video.")
            
            # Reset button
            if st.button("🔄 Analyze Another Video", use_container_width=True):
                st.session_state.processing_complete = False
                st.session_state.prediction_result = None
                st.session_state.transcript_text = ""
                st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            HybridDeepfakeDetector v1.0 | Built with Streamlit and PyTorch
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""
    try:
        app = DeepfakeDetectorApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
