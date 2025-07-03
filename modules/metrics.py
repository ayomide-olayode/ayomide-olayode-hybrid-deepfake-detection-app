import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple, Any

class MetricsCalculator:
    """Calculate various performance metrics for deepfake detection"""
    
    def __init__(self):
        self.ground_truth = []
        self.predictions = []
        self.confidences = []
        self.frame_predictions = []
        self.modality_embeddings = {'video': [], 'audio': [], 'text': []}
    
    def add_sample(self, ground_truth: int, prediction: int, confidence: float):
        """Add a sample for aggregate metrics"""
        self.ground_truth.append(ground_truth)
        self.predictions.append(prediction)
        self.confidences.append(confidence)
    
    def add_frame_predictions(self, frame_predictions: List[float]):
        """Add frame-level predictions for temporal analysis"""
        self.frame_predictions.append(frame_predictions)
    
    def add_modality_embeddings(self, video_emb: np.ndarray, audio_emb: np.ndarray, text_emb: np.ndarray):
        """Add embeddings for modality alignment calculation"""
        if video_emb is not None and len(video_emb) > 0:
            self.modality_embeddings['video'].append(video_emb)
        if audio_emb is not None and len(audio_emb) > 0:
            self.modality_embeddings['audio'].append(audio_emb)
        if text_emb is not None and len(text_emb) > 0:
            self.modality_embeddings['text'].append(text_emb)
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all available metrics"""
        metrics = {}
        
        # Basic classification metrics
        if len(self.ground_truth) > 0:
            metrics['accuracy'] = accuracy_score(self.ground_truth, self.predictions)
            metrics['precision'] = precision_score(self.ground_truth, self.predictions)
            metrics['recall'] = recall_score(self.ground_truth, self.predictions)
            metrics['f1_score'] = f1_score(self.ground_truth, self.predictions)
            
            # ROC-AUC needs probabilities
            if len(set(self.ground_truth)) >= 2:  # Need both classes
                metrics['roc_auc'] = roc_auc_score(self.ground_truth, self.confidences)
            
            # Confusion matrix
            cm = confusion_matrix(self.ground_truth, self.predictions)
            metrics['confusion_matrix'] = {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            }
        
        # Frame-wise detection rate
        if len(self.frame_predictions) > 0:
            frame_detection_rates = []
            for preds in self.frame_predictions:
                if len(preds) > 0:
                    frame_detection_rates.append(np.mean(preds))
            metrics['frame_detection_rate'] = np.mean(frame_detection_rates) if frame_detection_rates else 0
        
        # Modality alignment score
        metrics['modality_alignment'] = self.calculate_modality_alignment()
        
        return metrics
    
    def calculate_modality_alignment(self) -> Dict[str, float]:
        """Calculate alignment scores between different modalities"""
        alignment_scores = {}
        
        # Video-Audio alignment
        if self.modality_embeddings['video'] and self.modality_embeddings['audio']:
            min_len = min(len(self.modality_embeddings['video']), len(self.modality_embeddings['audio']))
            video_embs = np.array(self.modality_embeddings['video'][:min_len])
            audio_embs = np.array(self.modality_embeddings['audio'][:min_len])
            alignment_scores['video_audio'] = self._cosine_similarity(video_embs, audio_embs)
        
        # Video-Text alignment
        if self.modality_embeddings['video'] and self.modality_embeddings['text']:
            min_len = min(len(self.modality_embeddings['video']), len(self.modality_embeddings['text']))
            video_embs = np.array(self.modality_embeddings['video'][:min_len])
            text_embs = np.array(self.modality_embeddings['text'][:min_len])
            alignment_scores['video_text'] = self._cosine_similarity(video_embs, text_embs)
        
        # Audio-Text alignment
        if self.modality_embeddings['audio'] and self.modality_embeddings['text']:
            min_len = min(len(self.modality_embeddings['audio']), len(self.modality_embeddings['text']))
            audio_embs = np.array(self.modality_embeddings['audio'][:min_len])
            text_embs = np.array(self.modality_embeddings['text'][:min_len])
            alignment_scores['audio_text'] = self._cosine_similarity(audio_embs, text_embs)
        
        return alignment_scores
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate average cosine similarity between two sets of embeddings"""
        if emb1.shape != emb2.shape or len(emb1) == 0:
            return 0.0
        
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        cosine_sim = np.sum(emb1_norm * emb2_norm, axis=1)
        return float(np.mean(cosine_sim))