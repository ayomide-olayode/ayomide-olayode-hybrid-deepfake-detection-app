"""
Training script for the HybridDeepfakeDetector.
Handles model training with logging, checkpointing, evaluation, plotting and proper resume.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import config
from models.hybrid_model import create_model, get_model_config
from datasets.deepfake_dataset import create_data_loaders, get_dataset_config
from utils.model_utils import (
    ModelCheckpoint,
    MetricsCalculator,
    EarlyStopping,
    print_model_info,
    save_training_history,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for the hybrid deepfake detector."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 10,
        checkpoint_dir: str = None,
    ):
        """
        Initialize trainer.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Use BCEWithLogitsLoss (expects raw logits)
        self.criterion = nn.BCEWithLogitsLoss()

        # Utilities
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir or str(config.MODEL_OUTPUT_DIR),
            model_name="hybrid_deepfake_detector",
        )

        self.early_stopping = EarlyStopping(
            patience=patience, mode="max", restore_best_weights=True
        )

        self.metrics_calculator = MetricsCalculator()

        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rates": [],
        }

        # LR scheduler (robust to torch versions)
        try:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5
            )

        logger.info("Trainer initialized successfully")
        print_model_info(self.model, "HybridDeepfakeDetector")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()

        running_loss = 0.0
        all_prob_predictions = []  # probabilities for AUC
        all_binary_preds = []  # binary preds at 0.5
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, (frames, text_data, labels, metadata) in enumerate(pbar):
            frames = frames.to(self.device)
            text_data = {k: v.to(self.device) for k, v in text_data.items()}
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            try:
                logits = self.model(frames, text_data)
                logits = logits.view(-1)
                labels = labels.float().view(-1)

                loss = self.criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                binary_preds = (probs >= 0.5).astype(int)

                all_prob_predictions.extend(probs)
                all_binary_preds.extend(binary_preds)
                all_labels.extend(labels.detach().cpu().numpy())

                pbar.set_postfix(
                    {"Loss": f"{loss.item():.4f}", "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}"}
                )

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue

        avg_loss = running_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0

        all_prob_predictions = np.array(all_prob_predictions)
        all_binary_preds = np.array(all_binary_preds)
        all_labels = np.array(all_labels)

        metrics = self.metrics_calculator.calculate_binary_metrics(
            y_true=all_labels, y_pred=all_binary_preds, y_prob=all_prob_predictions
        )
        metrics["loss"] = avg_loss

        return metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()

        running_loss = 0.0
        all_prob_predictions = []
        all_binary_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch_idx, (frames, text_data, labels, metadata) in enumerate(pbar):
                frames = frames.to(self.device)
                text_data = {k: v.to(self.device) for k, v in text_data.items()}
                labels = labels.to(self.device)

                try:
                    logits = self.model(frames, text_data)
                    logits = logits.view(-1)
                    labels = labels.float().view(-1)

                    loss = self.criterion(logits, labels)
                    running_loss += loss.item()

                    probs = torch.sigmoid(logits).cpu().numpy()
                    binary_preds = (probs >= 0.5).astype(int)

                    all_prob_predictions.extend(probs)
                    all_binary_preds.extend(binary_preds)
                    all_labels.extend(labels.cpu().numpy())

                    pbar.set_postfix(
                        {"Loss": f"{loss.item():.4f}", "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}"}
                    )

                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

        avg_loss = running_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0

        all_prob_predictions = np.array(all_prob_predictions)
        all_binary_preds = np.array(all_binary_preds)
        all_labels = np.array(all_labels)

        metrics = self.metrics_calculator.calculate_binary_metrics(
            y_true=all_labels, y_pred=all_binary_preds, y_prob=all_prob_predictions
        )
        metrics["loss"] = avg_loss

        return metrics

    def _plot_metrics(self, output_dir: Path):
        """
        Generate and save plots of training/validation metrics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        epochs = list(range(1, len(self.history["train_loss"]) + 1))
        if len(epochs) == 0:
            logger.warning("No epochs recorded; skipping plots.")
            return

        # Loss
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "loss.png")
        plt.close()

        # Accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history["train_accuracy"], label="Train Acc")
        plt.plot(epochs, self.history["val_accuracy"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy.png")
        plt.close()

        # F1
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history["train_f1"], label="Train F1")
        plt.plot(epochs, self.history["val_f1"], label="Val F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Training and Validation F1 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "f1.png")
        plt.close()

        # LR
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.history["learning_rates"], label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "lr.png")
        plt.close()

        logger.info(f"Training plots saved to {output_dir}")

    def train(self, num_epochs: int, start_epoch: int = 0) -> Dict[str, list]:
        """
        Main training loop with proper resume support.
        start_epoch: zero-based index to start from (e.g. 1 resumes at epoch 2)
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        best_f1 = 0.0
        start_time = time.time()

        # ensure start_epoch is within bounds
        if start_epoch < 0:
            start_epoch = 0
        if start_epoch >= num_epochs:
            logger.warning(f"start_epoch ({start_epoch}) >= num_epochs ({num_epochs}). Nothing to do.")
            return self.history

        for epoch in range(start_epoch, num_epochs):
            actual_epoch = epoch  # zero-based
            epoch_display = actual_epoch + 1  # one-based for logging/files

            epoch_start_time = time.time()

            logger.info(f"\nEpoch {epoch_display}/{num_epochs}")
            logger.info("-" * 50)

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            # Scheduler expects the monitored metric
            self.scheduler.step(val_metrics.get("f1", 0.0))

            # Save history
            self.history["train_loss"].append(train_metrics.get("loss", 0.0))
            self.history["train_accuracy"].append(train_metrics.get("accuracy", 0.0))
            self.history["train_f1"].append(train_metrics.get("f1", 0.0))
            self.history["val_loss"].append(val_metrics.get("loss", 0.0))
            self.history["val_accuracy"].append(val_metrics.get("accuracy", 0.0))
            self.history["val_f1"].append(val_metrics.get("f1", 0.0))
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            logger.info(f"Epoch {epoch_display} completed in {time.time() - epoch_start_time:.2f}s")
            self.metrics_calculator.print_metrics(train_metrics, "Train")
            self.metrics_calculator.print_metrics(val_metrics, "Validation")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")

            is_best = val_metrics.get("f1", 0.0) > best_f1
            if is_best:
                best_f1 = val_metrics.get("f1", 0.0)
                logger.info(f"New best F1 score: {best_f1:.4f}")

            # Save checkpoint with proper epoch number
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch_display,
                loss=val_metrics.get("loss", 0.0),
                metrics=val_metrics,
                is_best=is_best,
            )

            if self.early_stopping(val_metrics.get("f1", 0.0), self.model):
                logger.info(f"Early stopping triggered after epoch {epoch_display}")
                break

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time:.2f}s")
        logger.info(f"Best validation F1 score: {best_f1:.4f}")

        # Save history and plots
        history_path = Path(config.MODEL_OUTPUT_DIR) / "training_history.json"
        save_training_history(self.history, str(history_path))
        try:
            self._plot_metrics(Path(config.MODEL_OUTPUT_DIR))
        except Exception as e:
            logger.error(f"Could not generate training plots: {e}")

        return self.history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train hybrid deepfake detector")

    parser.add_argument(
        "--data_dir", type=str, default=str(config.DATA_DIR / "processed"), help="Directory containing processed data"
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--freeze_visual", action="store_true", help="Freeze visual backbone during training")
    parser.add_argument("--freeze_text", action="store_true", help="Freeze text backbone during training")

    args = parser.parse_args()

    config.print_config()
    logger.info(f"Training arguments: {vars(args)}")

    config.create_directories()

    device = config.DEVICE
    logger.info(f"Using device: {device}")

    model_config = get_model_config()
    model_config["freeze_visual_backbone"] = args.freeze_visual
    model_config["freeze_text_backbone"] = args.freeze_text

    model = create_model(model_config)

    dataset_config = get_dataset_config()
    dataset_config["batch_size"] = args.batch_size
    # for Windows CPU runs, prefer single-worker loading by default
    dataset_config.setdefault("num_workers", 0)

    try:
        train_loader, val_loader = create_data_loaders(data_dir=args.data_dir, **dataset_config)
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        logger.error("Make sure to run preprocessing first to create processed data")
        return

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        patience=args.patience,
        checkpoint_dir=str(config.MODEL_OUTPUT_DIR),
    )

    start_epoch = 0
    checkpoint = None
    if args.resume:
        try:
            checkpoint = trainer.checkpoint_manager.load_checkpoint(
                model=trainer.model, optimizer=trainer.optimizer, checkpoint_path=args.resume
            )
            saved_epoch = checkpoint.get("epoch", 0)
            # If checkpoint stores last completed epoch N, resume from next epoch index N
            if isinstance(saved_epoch, int) and saved_epoch > 0:
                start_epoch = saved_epoch  # start at saved_epoch (zero-based for loop)
                logger.info(f"Resuming from saved epoch {saved_epoch}")
            else:
                logger.info(f"Resumed training from checkpoint, starting from epoch {start_epoch + 1}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return

    try:
        history = trainer.train(args.epochs, start_epoch)
        logger.info("Training completed successfully!")

        if history["val_f1"]:
            best_val_f1 = max(history["val_f1"])
            best_epoch = history["val_f1"].index(best_val_f1) + 1
            logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
# EOF