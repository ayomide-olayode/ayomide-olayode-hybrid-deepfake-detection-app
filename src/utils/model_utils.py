"""
Model utility functions for the HybridDeepfakeDetector project.
Handles model saving, loading, and evaluation utilities.
"""

import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_torch_save(obj, path):
    """
    Save `obj` atomically to `path`.
    Writes to a temp file in the same directory then atomically replaces target.
    Falls back to the old zipfile serialization if the default fails.
    """
    path = os.fspath(path)
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_ckpt_", dir=dirpath)
    os.close(fd)
    try:
        try:
            # Primary save (new zipfile serialization)
            torch.save(obj, tmp_path)
        except Exception as e_new:
            logger.warning(f"Primary torch.save failed: {e_new}. Trying fallback serializer.")
            # Fallback to older serializer
            torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)

        # Best-effort flush: open and fsync
        try:
            with open(tmp_path, "rb") as f:
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # ignore sync issues
                    pass
        except Exception:
            pass

        # Atomic replace
        os.replace(tmp_path, path)
        logger.info(f"Checkpoint saved atomically: {path}")

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

class ModelCheckpoint:
    """Handles model checkpointing and saving/loading."""
    def __init__(self, checkpoint_dir: str, model_name: str = "hybrid_deepfake_detector"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_score = float('-inf')

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'loss': loss,
            'metrics': metrics,
            'model_name': self.model_name
        }
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pth"

        try:
            safe_torch_save(checkpoint, checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
            # Attempt alternate location as a fallback
            try:
                alt_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}_failed_{int(torch.randint(0, 1_000_000, (1,)).item())}.pth"
                logger.info(f"Attempting to save to alternate path {alt_path}")
                safe_torch_save(checkpoint, alt_path)
                checkpoint_path = alt_path
            except Exception as e2:
                logger.error(f"Alternate save also failed: {e2}")
                raise

        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            try:
                safe_torch_save(checkpoint, best_path)
                logger.info(f"New best model saved with score: {metrics.get('f1', loss)}")
            except Exception as e:
                logger.error(f"Failed to save best checkpoint to {best_path}: {e}")
                # best save failure shouldn't prevent training continuation; continue

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                       checkpoint_path: Optional[str] = None, load_best: bool = True) -> Dict[str, Any]:
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            else:
                checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_epoch_*.pth"))
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
                checkpoint_path = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))

        checkpoint_path = str(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                # optimizer state might be incompatible across different optimizers or versions
                logger.warning("Could not load optimizer state dict (incompatible shapes/keys)")

        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        return checkpoint

class MetricsCalculator:
    """Calculates and tracks various evaluation metrics."""

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    @staticmethod
    def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None,
                                 threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate binary classification metrics.

        - y_true: true labels (0/1)
        - y_pred: either binary predictions (0/1) OR probabilities (0..1)
        - y_prob: explicit probabilities for AUC (preferred if available)
        - threshold: used to turn probabilities into binary preds if needed
        """
        y_true = MetricsCalculator._to_numpy(y_true).astype(int)

        # If y_prob provided use it. Otherwise, if y_pred looks like probabilities (floats in [0,1]) use them.
        if y_prob is not None:
            y_prob = MetricsCalculator._to_numpy(y_prob).astype(float)
        else:
            y_pred_np = MetricsCalculator._to_numpy(y_pred)
            if y_pred_np.dtype.kind == 'f' and (y_pred_np.min() >= 0.0 and y_pred_np.max() <= 1.0):
                y_prob = y_pred_np.astype(float)
            else:
                y_prob = None

        # Now determine binary predictions to compute accuracy/precision/recall/f1
        y_pred_np = MetricsCalculator._to_numpy(y_pred)
        # If y_pred is floats in 0..1, threshold them
        if y_pred_np.dtype.kind == 'f' and (y_pred_np.min() >= 0.0 and y_pred_np.max() <= 1.0):
            y_pred_binary = (y_pred_np >= threshold).astype(int)
        else:
            # Expect y_pred already 0/1 integers
            y_pred_binary = y_pred_np.astype(int)

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred_binary)),
            'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred_binary, zero_division=0))
        }

        # AUC calculation: prefer y_prob, then try y_pred if in [0,1]
        auc_val = None
        if y_prob is not None:
            try:
                auc_val = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auc_val = None
        else:
            # fallback: try y_pred if it looks like probabilities
            y_pred_check = MetricsCalculator._to_numpy(y_pred)
            if y_pred_check.dtype.kind == 'f' and y_pred_check.min() >= 0.0 and y_pred_check.max() <= 1.0:
                try:
                    auc_val = float(roc_auc_score(y_true, y_pred_check))
                except Exception:
                    auc_val = None

        metrics['auc'] = float(auc_val) if auc_val is not None else 0.0

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):
        print(f"\n{prefix} Metrics:")
        print("-" * 40)
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'loss']:
            if metric_name in metrics:
                print(f"{metric_name.capitalize()}: {metrics[metric_name]:.4f}")
        # print extra metrics if any
        extras = {k: v for k, v in metrics.items() if k not in ['accuracy','precision','recall','f1','auc','loss']}
        for k, v in extras.items():
            print(f"{k}: {v}")
        print("-" * 40)

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'max', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                # copy state dict
                self.best_weights = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
            return True
        return False

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_info(model: nn.Module, model_name: str = "Model"):
    total_params, trainable_params = count_parameters(model)
    print(f"\n{model_name} Information:")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 50)

def save_training_history(history: Dict[str, list], save_path: str):
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {save_path}")

def load_training_history(load_path: str) -> Dict[str, list]:
    with open(load_path, 'r') as f:
        history = json.load(f)
    logger.info(f"Training history loaded from {load_path}")
    return history
