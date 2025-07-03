import torch
import numpy as np
from typing import Dict, List, Any
import tempfile
import subprocess
import os
import logging
import soundfile as sf
import librosa
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing module using locally stored Whisper model and audio analysis"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.model_cache = "model_cache/whisper"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Find FFmpeg executable
        self.ffmpeg_path = self.find_ffmpeg()
        if not self.ffmpeg_path:
            logger.error("FFmpeg not found! Install with: winget install ffmpeg")
        
        # Load Whisper model from local cache
        logger.info("Loading Whisper model from local cache...")
        try:
            # Check if model files exist
            required_files = [
                "config.json", "preprocessor_config.json", 
                "tokenizer_config.json", "vocab.json",
                "tokenizer.json", "pytorch_model.bin",
                "merges.txt", "special_tokens_map.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(self.model_cache, file)):
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(
                    f"Missing Whisper model files: {', '.join(missing_files)}"
                )
            
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            self.processor = WhisperProcessor.from_pretrained(
                self.model_cache, 
                local_files_only=True
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_cache, 
                local_files_only=True
            ).to(self.device)
            
            logger.info("Whisper model loaded successfully from local cache")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None
            self.processor = None
    
    def find_ffmpeg(self) -> str:
        """Find FFmpeg executable in common locations"""
        # Check system PATH first
        if shutil.which("ffmpeg"):
            return "ffmpeg"
            
        # Check common Windows installation paths
        common_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
                
        return None
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file using FFmpeg"""
        if not self.ffmpeg_path:
            logger.error("FFmpeg not available!")
            return None
            
        try:
            # Create temporary audio file
            audio_path = tempfile.mktemp(suffix='.wav')
            
            # Use ffmpeg to extract audio
            cmd = [
                self.ffmpeg_path, '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate), '-ac', '1', audio_path, '-y'
            ]
            
            # Run FFmpeg with error handling
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60  # Increased timeout
            )
            
            if result.returncode == 0 and os.path.exists(audio_path):
                logger.info(f"Audio extracted to: {audio_path}")
                return audio_path
            else:
                logger.error(f"FFmpeg failed with error: {result.stderr.decode('utf-8')}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using locally loaded Whisper model"""
        if self.model is None or self.processor is None:
            return {
                'text': "Whisper model not available",
                'segments': [],
                'language': 'unknown',
                'confidence': 0.0,
                'segment_embeddings': []
            }
        
        try:
            logger.info("Transcribing audio with local Whisper model...")
            
            # Load audio file
            audio_data, sr = sf.read(audio_path)
            
            # Process audio
            input_features = self.processor(
                audio_data, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Get encoder outputs (audio embeddings)
            with torch.no_grad():
                encoder_outputs = self.model.model.encoder(input_features)
                embeddings = encoder_outputs.last_hidden_state
                segment_embedding = embeddings.mean(dim=1).cpu().numpy()[0]
            
            # Generate transcription
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence based on transcription length
            confidence = min(len(transcription) / 1000.0, 0.95)  # Scale confidence
            
            logger.info(f"Transcription completed. Text length: {len(transcription)}")
            
            return {
                'text': transcription,
                'segments': [],  # Not implemented in this simplified version
                'language': 'english',  # Assuming English
                'confidence': confidence,
                'segment_embeddings': [segment_embedding]  # Return as list with one segment
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                'text': f"Transcription failed: {str(e)}",
                'segments': [],
                'language': 'unknown',
                'confidence': 0.0,
                'segment_embeddings': []
            }
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract comprehensive audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=10.0)
            
            if len(y) == 0:
                raise ValueError("Empty audio file")
            
            logger.info(f"Extracting features from audio: {len(y)} samples, {sr} Hz")
            
            # Extract features
            features = {}
            
            # Spectral features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
            features['chroma_std'] = np.std(chroma, axis=1).tolist()
            
            # Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = float(tempo)
            except:
                features['tempo'] = 120.0
            
            logger.info("Audio features extracted successfully")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {'error': str(e)}
    
    def detect_audio_artifacts(self, audio_path: str) -> Dict[str, float]:
        """Detect audio artifacts that might indicate manipulation"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=10.0)
            
            if len(y) == 0:
                raise ValueError("Empty audio file")
            
            artifacts = {}
            
            # Spectral irregularities
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Calculate spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            artifacts['spectral_flatness_anomaly'] = float(np.std(spectral_flatness))
            
            # Detect sudden amplitude changes
            rms = librosa.feature.rms(y=y)[0]
            rms_diff = np.diff(rms)
            artifacts['amplitude_discontinuities'] = float(np.std(rms_diff))
            
            # Frequency domain analysis
            fft = np.fft.fft(y)
            magnitude_spectrum = np.abs(fft)
            
            # Look for unnatural frequency patterns
            freq_bins = np.fft.fftfreq(len(fft), 1/sr)
            positive_freqs = freq_bins[:len(freq_bins)//2]
            positive_magnitude = magnitude_spectrum[:len(magnitude_spectrum)//2]
            
            # Calculate spectral entropy
            normalized_magnitude = positive_magnitude / np.sum(positive_magnitude)
            spectral_entropy = -np.sum(normalized_magnitude * np.log(normalized_magnitude + 1e-10))
            artifacts['spectral_entropy'] = float(spectral_entropy / np.log(len(positive_magnitude)))
            
            # Detect clipping
            clipping_threshold = 0.95 * np.max(np.abs(y))
            clipped_samples = np.sum(np.abs(y) > clipping_threshold)
            artifacts['clipping_ratio'] = float(clipped_samples / len(y))
            
            # Normalize artifact scores to 0-1 range
            for key in artifacts:
                artifacts[key] = min(artifacts[key], 1.0)
            
            logger.info("Audio artifacts detected successfully")
            return artifacts
            
        except Exception as e:
            logger.error(f"Error detecting audio artifacts: {e}")
            return {
                'spectral_flatness_anomaly': 0.5,
                'amplitude_discontinuities': 0.5,
                'spectral_entropy': 0.5,
                'clipping_ratio': 0.5,
                'error': str(e)
            }
    
    def analyze_speech_naturalness(self, transcription: Dict[str, Any]) -> Dict[str, float]:
        """Analyze speech naturalness from transcription"""
        try:
            text = transcription.get('text', '').strip()
            segments = transcription.get('segments', [])
            
            if not text:
                return {'naturalness_score': 0.5, 'speech_rate': 0.0}
            
            # Calculate speech rate (words per minute)
            words = text.split()
            word_count = len(words)
            
            # Estimate duration based on word count if no segments
            if segments:
                total_duration = segments[-1].get('end', 0) - segments[0].get('start', 0)
            else:
                # Estimate 0.3 seconds per word
                total_duration = word_count * 0.3
            
            if total_duration > 0:
                speech_rate = word_count / (total_duration / 60.0)
            else:
                speech_rate = 150.0  # Default speech rate
            
            # Analyze naturalness based on speech patterns
            naturalness_score = 0.8  # Base score
            
            # Penalize very fast or very slow speech
            if speech_rate < 80 or speech_rate > 250:
                naturalness_score -= 0.2
            
            # Check for repetitive patterns
            if word_count > 0:
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                max_freq = max(word_freq.values())
                repetition_ratio = max_freq / word_count
                if repetition_ratio > 0.3:  # Too repetitive
                    naturalness_score -= 0.3
            else:
                repetition_ratio = 0.0
            
            return {
                'naturalness_score': max(0.0, min(1.0, naturalness_score)),
                'speech_rate': speech_rate,
                'word_count': word_count,
                'repetition_ratio': repetition_ratio
            }
            
        except Exception as e:
            logger.error(f"Error analyzing speech naturalness: {e}")
            return {
                'naturalness_score': 0.5,
                'speech_rate': 150.0,
                'word_count': 0,
                'error': str(e)
            }
    
    def process(self, video_path: str) -> Dict[str, Any]:
        """Main processing function"""
        try:
            logger.info(f"Processing audio from video: {video_path}")
            
            # Extract audio from video
            audio_path = self.extract_audio(video_path)
            
            if audio_path is None:
                raise ValueError("Failed to extract audio from video")
            
            try:
                # Transcribe audio
                transcription = self.transcribe_audio(audio_path)
                
                # Get segment embeddings from transcription
                segment_embeddings = transcription.get('segment_embeddings', [])
                
                # Extract audio features
                audio_features = self.extract_audio_features(audio_path)
                
                # Detect artifacts
                artifacts = self.detect_audio_artifacts(audio_path)
                
                # Analyze speech naturalness
                speech_analysis = self.analyze_speech_naturalness(transcription)
                
                # Calculate overall confidence
                artifact_score = np.mean(list(artifacts.values()))
                naturalness_score = speech_analysis.get('naturalness_score', 0.5)
                transcription_confidence = transcription.get('confidence', 0.5)
                
                # Combine scores
                overall_confidence = (
                    artifact_score * 0.4 +
                    (1 - naturalness_score) * 0.3 +
                    (1 - transcription_confidence) * 0.3
                )
                
                result = {
                    'confidence': float(overall_confidence),
                    'transcription': transcription,
                    'audio_features': audio_features,
                    'artifacts': artifacts,
                    'speech_analysis': speech_analysis,
                    'segment_embeddings': segment_embeddings,
                    'model_info': {
                        'whisper_model': 'base',
                        'sample_rate': self.sample_rate,
                        'max_duration': 10.0
                    }
                }
                
                logger.info("Audio processing completed successfully")
                return result
                
            finally:
                # Clean up temporary audio file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                'confidence': 0.5,
                'error': str(e),
                'transcription': {'text': 'Error in transcription'},
                'audio_features': {},
                'artifacts': {},
                'speech_analysis': {},
                'segment_embeddings': [],
                'model_info': {
                    'whisper_model': 'base',
                    'error': str(e)
                }
            }