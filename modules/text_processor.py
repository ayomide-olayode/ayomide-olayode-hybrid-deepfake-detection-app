import torch
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
import cv2
import numpy as np
from typing import Dict, List, Any
import re
import logging
from PIL import Image
import pytesseract
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing module using Tesseract OCR, BERT, and CLIP with local models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Configure Tesseract path
        if os.name == 'nt':
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info("Tesseract configured for Windows")
            else:
                logger.warning("Tesseract not found at default Windows path")
        else:
            pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
            logger.info("Tesseract configured for Linux/Mac")
        
        # Initialize BERT
        logger.info("Loading BERT model from local cache...")
        try:
            bert_path = "model_cache/bert"
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                bert_path, 
                local_files_only=True
            )
            self.bert_model = BertModel.from_pretrained(
                bert_path, 
                local_files_only=True
            ).to(self.device)
            self.bert_model.eval()
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
        
        # Initialize CLIP
        logger.info("Loading CLIP model from local cache...")
        try:
            clip_path = "model_cache/clip"
            
            # Verify required files exist
            required_files = [
                "config.json", "preprocessor_config.json", "pytorch_model.bin",
                "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"
            ]
            
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(clip_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(
                    f"Missing CLIP files: {', '.join(missing_files)}"
                )
            
            self.clip_processor = CLIPProcessor.from_pretrained(
                clip_path, 
                local_files_only=True
            )
            self.clip_model = CLIPModel.from_pretrained(
                clip_path, 
                local_files_only=True
            ).to(self.device)
            self.clip_model.eval()
            logger.info("CLIP model loaded successfully")
            
            # Set max sequence length for CLIP
            self.CLIP_MAX_SEQ_LENGTH = 77
            logger.info(f"CLIP max sequence length: {self.CLIP_MAX_SEQ_LENGTH}")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.clip_processor = None
            self.clip_model = None
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR results"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
            return blurred
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def extract_frames_from_video(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from video for text analysis"""
        cap = None
        try:
            # Convert to absolute path to avoid permission issues
            video_path = os.path.abspath(video_path)
            logger.info(f"Opening video: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // max_frames)
            
            # Extract frames
            frames = []
            frame_count = 0
            
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on interval
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                
                frame_count += 1
            
            logger.info(f"Extracted {len(frames)} frames for text analysis")
            return frames
        
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
        
        finally:
            if cap and cap.isOpened():
                cap.release()
    
    def extract_text_from_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Extract text from video frames using Tesseract OCR"""
        try:
            all_texts = []
            frame_texts = []
            
            for i, frame in enumerate(frames):
                try:
                    processed_frame = self.preprocess_image_for_ocr(frame)
                    text = pytesseract.image_to_string(
                        processed_frame, 
                        lang='eng',
                        config='--psm 6 --oem 3'
                    )
                    cleaned_text = re.sub(r'\s+', ' ', text).strip()
                    if cleaned_text:
                        frame_texts.append({
                            'frame_index': i,
                            'text': cleaned_text
                        })
                        all_texts.append(cleaned_text)
                    
                except Exception as e:
                    logger.warning(f"Error processing frame {i}: {e}")
                    frame_texts.append({
                        'frame_index': i,
                        'text': "",
                        'error': str(e)
                    })
            
            logger.info(f"OCR completed. Found {len(all_texts)} text instances")
            
            return {
                'all_texts': all_texts,
                'frame_texts': frame_texts,
                'total_text_instances': len(all_texts)
            }
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return {
                'all_texts': [],
                'frame_texts': [],
                'total_text_instances': 0,
                'error': str(e)
            }
    
    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """Get BERT embeddings for text segment"""
        if self.bert_model is None or not text.strip():
            return np.zeros(768, dtype=np.float32)
        
        try:
            inputs = self.bert_tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting BERT embeddings: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def get_clip_text_embeddings(self, text: str) -> np.ndarray:
        """Get CLIP text embeddings with truncation for long sequences"""
        if self.clip_model is None or not text.strip():
            return np.zeros(512, dtype=np.float32)
        
        try:
            # Truncate text to CLIP's max sequence length
            tokens = self.clip_processor.tokenizer.tokenize(text)
            if len(tokens) > self.CLIP_MAX_SEQ_LENGTH:
                truncated_text = self.clip_processor.tokenizer.convert_tokens_to_string(
                    tokens[:self.CLIP_MAX_SEQ_LENGTH]
                )
                logger.warning(f"Truncating text from {len(tokens)} to {self.CLIP_MAX_SEQ_LENGTH} tokens")
                text = truncated_text
            
            # Process text with truncation
            inputs = self.clip_processor(
                text=[text], 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=self.CLIP_MAX_SEQ_LENGTH
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                embeddings = text_features.cpu().numpy()[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting CLIP embeddings: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def analyze_semantic_consistency(self, text: str) -> Dict[str, float]:
        """Analyze semantic consistency using BERT"""
        if self.bert_model is None or not text.strip():
            return {
                'consistency_score': 0.5,
                'confidence': 0.5
            }
        
        try:
            embeddings = self.get_bert_embeddings(text)
            consistency_score = 1.0 / (1.0 + np.std(embeddings))
            confidence = min(consistency_score * 1.5, 1.0)
            
            return {
                'consistency_score': float(consistency_score),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error in BERT analysis: {e}")
            return {
                'consistency_score': 0.5,
                'confidence': 0.5,
                'error': str(e)
            }
    
    def detect_linguistic_anomalies(self, text: str) -> Dict[str, float]:
        """Detect linguistic anomalies in text with improved error handling"""
        if not text.strip():
            return {
                'repetitive_patterns': 0.0,
                'unnatural_transitions': 0.0,
                'vocabulary_inconsistencies': 0.0,
                'grammatical_errors': 0.0
            }
        
        try:
            anomalies = {}
            words = text.lower().split()
            word_count = len(words)
            
            # 1. Check for repetitive patterns
            if word_count > 0:
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                max_freq = max(word_freq.values())
                repetition_ratio = max_freq / word_count
                anomalies['repetitive_patterns'] = min(repetition_ratio * 2, 1.0)
            else:
                anomalies['repetitive_patterns'] = 0.0
            
            # 2. Vocabulary inconsistencies
            if word_count > 0:
                word_lengths = [len(word) for word in words]
                length_std = np.std(word_lengths)
                anomalies['vocabulary_inconsistencies'] = min(length_std / 5.0, 1.0)
            else:
                anomalies['vocabulary_inconsistencies'] = 0.0
            
            # 3. Grammatical error detection (with improved safety)
            grammar_score = 0.0
            
            # Split into sentences safely
            sentences = re.split(r'[.!?]+', text)
            valid_sentences = [s.strip() for s in sentences if s.strip()]
            
            if valid_sentences:
                # Check for capitalization of first sentence
                first_sentence = valid_sentences[0]
                if first_sentence and not first_sentence[0].isupper():
                    grammar_score += 0.2
                
                # Check for double spaces
                if '  ' in text:
                    grammar_score += 0.1
            else:
                # No valid sentences found
                logger.warning("No valid sentences for grammar analysis")
            
            anomalies['grammatical_errors'] = min(grammar_score, 1.0)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting linguistic anomalies: {e}")
            return {
                'repetitive_patterns': 0.5,
                'unnatural_transitions': 0.5,
                'vocabulary_inconsistencies': 0.5,
                'grammatical_errors': 0.5,
                'error': str(e)
            }
    
    def analyze_text_visual_consistency(self, frames: List[np.ndarray], texts: List[str]) -> Dict[str, float]:
        """Analyze consistency between text and visual content using CLIP"""
        if self.clip_model is None or not frames or not texts:
            return {
                'consistency_score': 0.5,
                'alignment_score': 0.5
            }
        
        try:
            # Limit text length to avoid sequence errors
            sample_texts = []
            for text in texts[:min(3, len(texts))]:
                tokens = self.clip_processor.tokenizer.tokenize(text)
                if len(tokens) > self.CLIP_MAX_SEQ_LENGTH:
                    truncated_text = self.clip_processor.tokenizer.convert_tokens_to_string(
                        tokens[:self.CLIP_MAX_SEQ_LENGTH]
                    )
                    sample_texts.append(truncated_text)
                else:
                    sample_texts.append(text)
            
            sample_frames = frames[:min(3, len(frames))]
            consistency_scores = []
            
            for frame in sample_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                for text in sample_texts:
                    if not text.strip():
                        continue
                        
                    try:
                        inputs = self.clip_processor(
                            text=[text], 
                            images=[frame_pil], 
                            return_tensors="pt", 
                            padding=True,
                            truncation=True,
                            max_length=self.CLIP_MAX_SEQ_LENGTH
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = self.clip_model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            similarity = torch.softmax(logits_per_image, dim=-1)
                            consistency_scores.append(similarity.cpu().numpy()[0][0])
                            
                    except Exception as e:
                        logger.warning(f"CLIP processing error: {e}")
                        consistency_scores.append(0.5)
            
            if consistency_scores:
                avg_consistency = np.mean(consistency_scores)
                alignment_score = 1 - min(np.std(consistency_scores), 1.0)
            else:
                avg_consistency = 0.5
                alignment_score = 0.5
            
            return {
                'consistency_score': float(avg_consistency),
                'alignment_score': float(alignment_score),
                'num_comparisons': len(consistency_scores)
            }
            
        except Exception as e:
            logger.error(f"Error in CLIP analysis: {e}")
            return {
                'consistency_score': 0.5,
                'alignment_score': 0.5,
                'error': str(e)
            }
    
    def compare_text_to_frames(self, text: str, frames: List[np.ndarray]) -> Dict[str, float]:
        """Enhanced text-video consistency check with truncation"""
        if self.clip_model is None or not text.strip() or not frames:
            return {'consistency_score': 0.5, 'frame_scores': []}
        
        try:
            # Truncate text to CLIP's max sequence length
            tokens = self.clip_processor.tokenizer.tokenize(text)
            if len(tokens) > self.CLIP_MAX_SEQ_LENGTH:
                text = self.clip_processor.tokenizer.convert_tokens_to_string(
                    tokens[:self.CLIP_MAX_SEQ_LENGTH]
                )
                logger.warning(f"Truncating transcription text to {self.CLIP_MAX_SEQ_LENGTH} tokens")
            
            frame_scores = []
            text_inputs = self.clip_processor(
                text=[text], 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=self.CLIP_MAX_SEQ_LENGTH
            ).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
            
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                image_inputs = self.clip_processor(
                    images=[frame_pil], 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**image_inputs)
                similarity = torch.nn.functional.cosine_similarity(
                    text_features, image_features
                )
                frame_scores.append(similarity.item())
            
            avg_score = np.mean(frame_scores) if frame_scores else 0.5
            return {
                'consistency_score': float(avg_score),
                'frame_scores': frame_scores
            }
            
        except Exception as e:
            logger.error(f"Text-frame comparison failed: {e}")
            return {'consistency_score': 0.5, 'frame_scores': []}
    
    def process(self, video_path: str, transcription_text: str = None) -> Dict[str, Any]:
        """Main processing function with captions/transcription prioritization"""
        try:
            logger.info(f"Processing text analysis for video: {video_path}")
            
            # Step 1: Extract frames from video
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                logger.warning("No frames extracted from video")
                frames = []
            
            # Step 2: Extract text from frames using OCR
            ocr_results = self.extract_text_from_frames(frames)
            all_texts = ocr_results.get('all_texts', [])
            
            # Determine text source priority
            has_captions = any(all_texts)
            use_transcription = not has_captions and transcription_text
            
            if use_transcription:
                logger.info("No captions detected - using audio transcription for text analysis")
                all_texts = [transcription_text]
                text_source = "transcription"
            else:
                logger.info(f"Using {'captions' if has_captions else 'audio transcription'} for text analysis")
                if transcription_text:
                    all_texts.append(transcription_text)
                text_source = "captions"
            
            # Step 3: Analyze semantic consistency
            semantic_analysis = self.analyze_semantic_consistency(" ".join(all_texts))
            
            # Step 4: Detect linguistic anomalies
            linguistic_anomalies = self.detect_linguistic_anomalies(" ".join(all_texts))
            
            # Step 5: Text-visual consistency
            if use_transcription:
                clip_analysis = self.compare_text_to_frames(transcription_text, frames)
            else:
                clip_analysis = self.analyze_text_visual_consistency(frames, all_texts)
            
            # Step 6: Get embeddings for each text segment
            segment_features = []
            
            # Process OCR segments
            if not use_transcription:
                for text in ocr_results.get('all_texts', []):
                    bert_emb = self.get_bert_embeddings(text)
                    clip_emb = self.get_clip_text_embeddings(text)
                    # Ensure both are float arrays before concatenation
                    if isinstance(bert_emb, np.ndarray) and isinstance(clip_emb, np.ndarray):
                        combined_emb = np.concatenate([bert_emb, clip_emb])
                        segment_features.append(combined_emb.astype(np.float32))
                    else:
                        logger.warning("Skipping embedding due to type mismatch")
            
            # Process transcription text
            if transcription_text:
                words = transcription_text.split()
                chunk_size = 20
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    bert_emb = self.get_bert_embeddings(chunk)
                    clip_emb = self.get_clip_text_embeddings(chunk)
                    if isinstance(bert_emb, np.ndarray) and isinstance(clip_emb, np.ndarray):
                        combined_emb = np.concatenate([bert_emb, clip_emb])
                        segment_features.append(combined_emb.astype(np.float32))
                    else:
                        logger.warning("Skipping transcription embedding due to type mismatch")
            
            # Step 7: Calculate overall confidence
            semantic_score = semantic_analysis.get('consistency_score', 0.5)
            anomaly_score = np.mean(list(linguistic_anomalies.values())) if linguistic_anomalies else 0.5
            clip_score = clip_analysis.get('consistency_score', 0.5)
            
            overall_confidence = (
                (1 - semantic_score) * 0.3 +
                anomaly_score * 0.4 +
                (1 - clip_score) * 0.3
            )
            overall_confidence = max(0.0, min(1.0, overall_confidence))
            
            return {
                'confidence': float(overall_confidence),
                'word_count': len(" ".join(all_texts).split()) if all_texts else 0,
                'ocr_results': ocr_results,
                'semantic_analysis': semantic_analysis,
                'linguistic_anomalies': linguistic_anomalies,
                'clip_analysis': clip_analysis,
                'segment_features': segment_features,
                'text_source': text_source,
                'has_captions': has_captions,
                'combined_text': " ".join(all_texts)[:500] + "..." if all_texts else "",
                'model_info': {
                    'ocr_engine': 'Tesseract',
                    'semantic_model': 'BERT-base-uncased',
                    'visual_text_model': 'CLIP-ViT-B/32',
                    'device': str(self.device)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {
                'confidence': 0.5,
                'word_count': 0,
                'error': str(e),
                'ocr_results': {'all_texts': [], 'frame_texts': [], 'total_text_instances': 0},
                'semantic_analysis': {'consistency_score': 0.5},
                'linguistic_anomalies': {},
                'clip_analysis': {'consistency_score': 0.5},
                'segment_features': [],
                'text_source': 'error',
                'has_captions': False,
                'model_info': {
                    'error': str(e)
                }
            }