"""
Validation script for the HybridDeepfakeDetector.
Evaluates trained models on test data with detailed metrics and analysis.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import config
from models.hybrid_model import create_model, get_model_config
from datasets.deepfake_dataset import create_data_loaders, get_dataset_config
from utils.model_utils import ModelCheckpoint, MetricsCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Validator:
    """Validator class for comprehensive model evaluation."""
    
    def __init__(self, 
                 model: nn.Module,
                 data_loader: DataLoader,
                 device: torch.device,
                 output_dir: str = None):
        """
        Initialize validator.
        
        Args:
            model: The trained model to evaluate
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            output_dir: Directory to save evaluation results
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else config.PREDICTIONS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
        
        logger.info("Validator initialized successfully")
    
    def evaluate(self, save_predictions: bool = True) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            save_predictions: Whether to save individual predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting model evaluation")
        
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_metadata = []
        
        # Progress bar for evaluation
        pbar = tqdm(self.data_loader, desc="Evaluating")
        
        with torch.no_grad():
            for batch_idx, (frames, text_data, labels, metadata) in enumerate(pbar):
                # Move data to device
                frames = frames.to(self.device)
                text_data = {k: v.to(self.device) for k, v in text_data.items()}
                labels = labels.to(self.device)
                
                try:
                    # Forward pass
                    probabilities = self.model(frames, text_data)
                    probabilities = probabilities.squeeze()
                    
                    # Convert probabilities to predictions (threshold = 0.5)
                    predictions = (probabilities >= 0.5).float()
                    
                    # Store results
                    all_predictions.extend(predictions.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Store metadata for each sample in batch
                    for i in range(len(metadata['video_name'])):
                        sample_metadata = {
                            'video_name': metadata['video_name'][i],
                            'class_name': metadata['class_name'][i],
                            'num_frames': metadata['num_frames'][i].item(),
                            'transcript_length': metadata['transcript_length'][i].item()
                        }
                        all_metadata.append(sample_metadata)
                    
                    # Update progress bar
                    batch_accuracy = (predictions == labels).float().mean().item()
                    pbar.set_postfix({'Batch Acc': f'{batch_accuracy:.4f}'})
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_binary_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities
        )
        
        # Generate detailed classification report
        class_names = ['Real', 'Fake']
        classification_rep = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            output_dict=True
        )
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Compile results
        results = {
            'metrics': metrics,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'total_samples': len(all_labels),
            'class_distribution': {
                'real': int(np.sum(all_labels == 0)),
                'fake': int(np.sum(all_labels == 1))
            }
        }
        
        # Save individual predictions if requested
        if save_predictions:
            predictions_data = []
            for i in range(len(all_predictions)):
                pred_data = {
                    'video_name': all_metadata[i]['video_name'],
                    'class_name': all_metadata[i]['class_name'],
                    'true_label': int(all_labels[i]),
                    'predicted_label': int(all_predictions[i]),
                    'probability': float(all_probabilities[i]),
                    'correct': bool(all_predictions[i] == all_labels[i]),
                    'num_frames': all_metadata[i]['num_frames'],
                    'transcript_length': all_metadata[i]['transcript_length']
                }
                predictions_data.append(pred_data)
            
            results['predictions'] = predictions_data
        
        # Print results
        self._print_results(results)
        
        # Save results
        self._save_results(results)
        
        # Create visualizations
        self._create_visualizations(results)
        
        return results
    
    def evaluate_ablation(self) -> Dict:
        """
        Perform ablation study (visual-only and text-only evaluation).
        
        Returns:
            Dictionary containing ablation study results
        """
        logger.info("Starting ablation study")
        
        self.model.eval()
        
        # Storage for different modalities
        results = {
            'visual_only': {'predictions': [], 'labels': []},
            'text_only': {'predictions': [], 'labels': []},
            'hybrid': {'predictions': [], 'labels': []}
        }
        
        with torch.no_grad():
            for frames, text_data, labels, metadata in tqdm(self.data_loader, desc="Ablation Study"):
                # Move data to device
                frames = frames.to(self.device)
                text_data = {k: v.to(self.device) for k, v in text_data.items()}
                labels = labels.to(self.device)
                
                try:
                    # Visual-only prediction
                    visual_pred = self.model.forward_visual_only(frames)
                    visual_pred = visual_pred.squeeze()
                    
                    # Text-only prediction
                    text_pred = self.model.forward_text_only(text_data)
                    text_pred = text_pred.squeeze()
                    
                    # Hybrid prediction
                    hybrid_pred = self.model(frames, text_data)
                    hybrid_pred = hybrid_pred.squeeze()
                    
                    # Store results
                    results['visual_only']['predictions'].extend(visual_pred.cpu().numpy())
                    results['text_only']['predictions'].extend(text_pred.cpu().numpy())
                    results['hybrid']['predictions'].extend(hybrid_pred.cpu().numpy())
                    
                    # Labels are the same for all modalities
                    labels_np = labels.cpu().numpy()
                    results['visual_only']['labels'].extend(labels_np)
                    results['text_only']['labels'].extend(labels_np)
                    results['hybrid']['labels'].extend(labels_np)
                    
                except Exception as e:
                    logger.error(f"Error in ablation study: {e}")
                    continue
        
        # Calculate metrics for each modality
        ablation_metrics = {}
        for modality in results:
            predictions = np.array(results[modality]['predictions'])
            labels = np.array(results[modality]['labels'])
            
            metrics = self.metrics_calculator.calculate_binary_metrics(
                y_true=labels,
                y_pred=predictions,
                y_prob=predictions
            )
            
            ablation_metrics[modality] = metrics
            
            logger.info(f"\n{modality.upper()} Results:")
            self.metrics_calculator.print_metrics(metrics)
        
        # Save ablation results
        ablation_path = self.output_dir / "ablation_study.json"
        with open(ablation_path, 'w') as f:
            json.dump(ablation_metrics, f, indent=2)
        
        logger.info(f"Ablation study results saved to {ablation_path}")
        
        return ablation_metrics
    
    def _print_results(self, results: Dict):
        """Print evaluation results."""
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        
        # Overall metrics
        self.metrics_calculator.print_metrics(results['metrics'], "Overall")
        
        # Class-wise metrics
        logger.info("\nClass-wise Performance:")
        logger.info("-" * 40)
        for class_name in ['Real', 'Fake']:
            class_key = class_name.lower()
            if class_key in results['classification_report']:
                class_metrics = results['classification_report'][class_key]
                logger.info(f"{class_name}:")
                logger.info(f"  Precision: {class_metrics['precision']:.4f}")
                logger.info(f"  Recall: {class_metrics['recall']:.4f}")
                logger.info(f"  F1-score: {class_metrics['f1-score']:.4f}")
                logger.info(f"  Support: {class_metrics['support']}")
        
        # Confusion matrix
        logger.info("\nConfusion Matrix:")
        logger.info("-" * 40)
        conf_matrix = np.array(results['confusion_matrix'])
        logger.info("           Predicted")
        logger.info("         Real  Fake")
        logger.info(f"Real   {conf_matrix[0,0]:6d} {conf_matrix[0,1]:5d}")
        logger.info(f"Fake   {conf_matrix[1,0]:6d} {conf_matrix[1,1]:5d}")
        
        # Class distribution
        logger.info(f"\nDataset Distribution:")
        logger.info(f"Real videos: {results['class_distribution']['real']}")
        logger.info(f"Fake videos: {results['class_distribution']['fake']}")
        logger.info(f"Total samples: {results['total_samples']}")
    
    def _save_results(self, results: Dict):
        """Save evaluation results to files."""
        # Save main results (without individual predictions to keep file size manageable)
        main_results = {k: v for k, v in results.items() if k != 'predictions'}
        
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(main_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        # Save individual predictions if available
        if 'predictions' in results:
            predictions_path = self.output_dir / "individual_predictions.json"
            with open(predictions_path, 'w') as f:
                json.dump(results['predictions'], f, indent=2)
            
            logger.info(f"Individual predictions saved to {predictions_path}")
    
    def _create_visualizations(self, results: Dict):
        """Create and save visualization plots."""
        try:
            # Confusion Matrix Heatmap
            plt.figure(figsize=(8, 6))
            conf_matrix = np.array(results['confusion_matrix'])
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Real', 'Fake'],
                       yticklabels=['Real', 'Fake'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            confusion_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix plot saved to {confusion_path}")
            
            # Metrics Bar Plot
            if 'predictions' in results:
                # Calculate per-class metrics for visualization
                metrics_data = results['classification_report']
                
                classes = ['Real', 'Fake']
                precision_scores = [metrics_data['real']['precision'], metrics_data['fake']['precision']]
                recall_scores = [metrics_data['real']['recall'], metrics_data['fake']['recall']]
                f1_scores = [metrics_data['real']['f1-score'], metrics_data['fake']['f1-score']]
                
                x = np.arange(len(classes))
                width = 0.25
                
                plt.figure(figsize=(10, 6))
                plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
                plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
                plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
                
                plt.xlabel('Classes')
                plt.ylabel('Score')
                plt.title('Per-Class Performance Metrics')
                plt.xticks(x, classes)
                plt.legend()
                plt.ylim(0, 1)
                
                # Add value labels on bars
                for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
                    plt.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom')
                    plt.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom')
                    plt.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom')
                
                metrics_plot_path = self.output_dir / "metrics_comparison.png"
                plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Metrics comparison plot saved to {metrics_plot_path}")
        
        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate hybrid deepfake detector")
    
    # TODO: Modify these default parameters based on your requirements
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR / "processed"),
                       help="Directory containing processed data")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                       help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default=str(config.PREDICTIONS_DIR),
                       help="Directory to save evaluation results")
    parser.add_argument("--subset", type=str, default="test", choices=["train", "test"],
                       help="Dataset subset to evaluate on")
    parser.add_argument("--ablation", action="store_true",
                       help="Perform ablation study")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                       help="Save individual predictions")
    
    args = parser.parse_args()
    
    # Print configuration
    config.print_config()
    logger.info(f"Validation arguments: {vars(args)}")
    
    # Set device
    device = config.DEVICE
    logger.info(f"Using device: {device}")
    
    # Create model
    model_config = get_model_config()
    model = create_model(model_config)
    
    # Load trained model
    if args.model_path:
        model_path = args.model_path
    else:
        # Use best model by default
        model_path = config.MODEL_OUTPUT_DIR / "hybrid_deepfake_detector_best.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        logger.error("Please train a model first or specify a valid model path")
        return
    
    # Load model checkpoint
    checkpoint_manager = ModelCheckpoint(str(config.MODEL_OUTPUT_DIR))
    try:
        checkpoint = checkpoint_manager.load_checkpoint(
            model=model,
            checkpoint_path=str(model_path)
        )
        logger.info(f"Loaded model from {model_path}")
        if 'metrics' in checkpoint:
            logger.info(f"Model training metrics: {checkpoint['metrics']}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create data loader for specified subset
    dataset_config = get_dataset_config()
    dataset_config['batch_size'] = args.batch_size
    
    try:
        if args.subset == "test":
            _, data_loader = create_data_loaders(data_dir=args.data_dir, **dataset_config)
        else:
            data_loader, _ = create_data_loaders(data_dir=args.data_dir, **dataset_config)
    except Exception as e:
        logger.error(f"Error creating data loader: {e}")
        logger.error("Make sure to run preprocessing first to create processed data")
        return
    
    # Create validator
    validator = Validator(
        model=model,
        data_loader=data_loader,
        device=device,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    try:
        results = validator.evaluate(save_predictions=args.save_predictions)
        
        # Run ablation study if requested
        if args.ablation:
            ablation_results = validator.evaluate_ablation()
        
        logger.info("Validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise

if __name__ == "__main__":
    main()
