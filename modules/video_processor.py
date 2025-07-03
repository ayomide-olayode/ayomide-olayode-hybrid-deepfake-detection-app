import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4
import numpy as np
from typing import Dict, List, Any
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processing module using locally stored EfficientNet-B4 for deepfake detection"""
    
    def __init__(self):
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load EfficientNet-B4 from local cache
        logger.info("Loading EfficientNet-B4 from local cache...")
        try:
            # 1. Create model architecture
            self.model = efficientnet_b4()
            
            # 2. Define path to local weights file
            model_dir = "model_cache/efficientnet"
            model_path = os.path.join(model_dir, "efficientnet_b4.pth")
            
            # 3. Verify weights file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model weights not found at {model_path}. "
                    f"Please download the weights to this location."
                )
            
            # 4. Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # 5. Modify classifier for binary classification
            # Get number of input features
            if isinstance(self.model.classifier, nn.Linear):
                num_features = self.model.classifier.in_features
            elif isinstance(self.model.classifier, nn.Sequential):
                for layer in self.model.classifier:
                    if isinstance(layer, nn.Linear):
                        num_features = layer.in_features
                        break
                else:
                    raise ValueError("No Linear layer found in classifier.")
            else:
                raise TypeError("Unexpected classifier type")
            
            # Replace classifier with custom binary classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 2),
                nn.Softmax(dim=1)
            )
            
            # 6. Move model to appropriate device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Video processor initialized successfully from local cache")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet: {e}")
            self.model = None
        
        # 7. Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),  # EfficientNet-B4 input size
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract frames from video file with efficient sampling"""
        cap = None
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video duration: {duration:.2f}s, FPS: {fps}, Total frames: {total_frames}")
            
            # Handle long videos
            max_duration = 30  # Maximum processing duration in seconds
            if duration > max_duration:
                max_frames_allowed = int(max_duration * fps)
                total_frames = min(total_frames, max_frames_allowed)
                logger.warning(f"Video duration exceeds {max_duration}s, limiting to first {max_frames_allowed} frames")
            
            # Calculate frame sampling interval
            frames = []
            frame_interval = max(1, total_frames // max_frames)
            
            # Extract frames with sampling
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frames and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on interval
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                
                frame_count += 1
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
        
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
        
        finally:
            # Ensure video capture is released
            if cap and cap.isOpened():
                cap.release()
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a single frame and return features + probabilities"""
        # Handle case where model failed to load
        if self.model is None:
            return {
                'real_probability': 0.5,
                'fake_probability': 0.5,
                'confidence': 0.5,
                'features': np.zeros(1792, dtype=np.float32)  # Actual EfficientNet-B4 feature dim
            }
        
        try:
            # Preprocess frame
            input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
            
            # Run inference to get features BEFORE classifier
            with torch.no_grad():
                # Get features from the layer before classifier
                features = self.model.features(input_tensor)
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
                
                # Get classification probabilities
                outputs = self.model.classifier(features)
                probabilities = outputs.cpu().numpy()[0]
                real_prob = probabilities[0]
                fake_prob = probabilities[1]
                
                # Convert features to numpy array
                features = features.cpu().numpy()[0]
            
            return {
                'real_probability': float(real_prob),
                'fake_probability': float(fake_prob),
                'confidence': float(max(real_prob, fake_prob)),
                'features': features  # Return the actual features (1792-dim)
            }
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {
                'real_probability': 0.5,
                'fake_probability': 0.5,
                'confidence': 0.5,
                'features': np.zeros(1792, dtype=np.float32)
            }
    
    def detect_compression_artifacts(self, frame: np.ndarray) -> float:
        """Detect compression artifacts using Laplacian variance"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate Laplacian variance (measure of image sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Higher variance = sharper image, lower compression
            # Normalize to 0-1 range (1000 is an empirical threshold)
            artifact_score = min(laplacian_var / 1000.0, 1.0)
            
            return artifact_score
        except Exception as e:
            logger.error(f"Error detecting compression artifacts: {e}")
            return 0.5
    
    def analyze_temporal_consistency(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Analyze consistency between consecutive frames using optical flow"""
        if len(frames) < 2:
            return {'temporal_consistency': 0.5, 'flow_magnitudes': []}
        
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
            flow_magnitudes = []
            
            for i in range(1, len(frames)):
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Compute magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_magnitudes.append(np.mean(magnitude))
                prev_gray = curr_gray
            
            # Calculate consistency based on flow magnitude changes
            magnitude_changes = np.abs(np.diff(flow_magnitudes))
            consistency_score = 1.0 - min(np.mean(magnitude_changes) / 5.0, 1.0)
            
            return {
                'temporal_consistency': consistency_score,
                'average_flow_magnitude': np.mean(flow_magnitudes),
                'flow_magnitudes': flow_magnitudes[:5]  # Return first few for debugging
            }
        except Exception as e:
            logger.error(f"Error analyzing temporal consistency: {e}")
            return {'temporal_consistency': 0.5, 'flow_magnitudes': []}
    
    def process(self, video_path: str) -> Dict[str, Any]:
        """Main processing pipeline for video analysis"""
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Step 1: Extract frames from video
            frames = self.extract_frames(video_path)
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Initialize result containers
            frame_results = []
            fake_probabilities = []
            compression_scores = []
            frame_features = []
            
            # Step 2: Process each frame
            for i, frame in enumerate(frames):
                # Analyze frame with EfficientNet
                frame_result = self.analyze_frame(frame)
                frame_results.append(frame_result)
                fake_probabilities.append(frame_result['fake_probability'])
                frame_features.append(frame_result['features'])
                
                # Detect compression artifacts
                compression_score = self.detect_compression_artifacts(frame)
                compression_scores.append(compression_score)
                
                logger.info(f"Processed frame {i+1}/{len(frames)}")
            
            # Step 3: Analyze temporal consistency
            temporal_analysis = self.analyze_temporal_consistency(frames)
            
            # Step 4: Calculate aggregated metrics
            avg_fake_prob = np.mean(fake_probabilities)
            max_fake_prob = np.max(fake_probabilities)
            std_fake_prob = np.std(fake_probabilities)
            avg_compression = np.mean(compression_scores)
            
            # Step 5: Calculate overall confidence
            confidence = (
                avg_fake_prob * 0.5 +
                max_fake_prob * 0.2 +
                (1 - temporal_analysis['temporal_consistency']) * 0.2 +
                avg_compression * 0.1
            )
            confidence = max(0.0, min(1.0, confidence))
            
            # Step 6: Compile results
            return {
                
                'confidence': float(confidence),
                'frames_analyzed': len(frames),
                'frame_features': frame_features,  # Actual frame embeddings
                'average_fake_probability': float(avg_fake_prob),
                'max_fake_probability': float(max_fake_prob),
                'std_fake_probability': float(std_fake_prob),
                'temporal_analysis': temporal_analysis,
                'compression_analysis': {
                    'average_compression_score': float(avg_compression),
                    'compression_scores': compression_scores[:5]
                },
                'frame_results': frame_results[:5],
                'model_info': {
                    'model_name': 'EfficientNet-B4',
                    'pretrained': True,
                    'device': str(self.device),
                    'status': 'success'
                },
                 'metrics': {
        'frame_fake_probs': fake_probabilities,
        'compression_scores': compression_scores,
        'temporal_scores': [temporal_analysis['temporal_consistency']],
        'feature_statistics': {
            'mean': float(np.mean(frame_features)),
            'std': float(np.std(frame_features)),
            'min': float(np.min(frame_features)),
            'max': float(np.max(frame_features))
        }
    }
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                'confidence': 0.5,
                'error': str(e),
                'frames_analyzed': 0,
                'frame_features': [],
                'model_info': {
                    'model_name': 'EfficientNet-B4',
                    'status': 'error',
                    'error': str(e)
                }
            }