"""
Training Utilities and Metrics for MobileViT Deepfake Detection

This module provides:
1. Complete training pipeline with all optimizations
2. Comprehensive metrics for evaluation
3. Model checkpointing and resuming
4. Mixed precision training for M4 Max
5. Early stopping and learning rate scheduling
6. TensorBoard logging and monitoring

All components are optimized for Apple Silicon (M4 Max) with MPS support.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


class MetricsCalculator:
    """
    Comprehensive metrics calculator for deepfake detection.

    This class computes various metrics that are important for evaluating
    deepfake detection models, including standard classification metrics
    and specialized metrics for binary classification tasks.
    """

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with new batch of predictions.

        Args:
            predictions: Model predictions (logits or class indices)
            targets: Ground truth labels
            probabilities: Softmax probabilities (optional)
        """
        # Convert to numpy for sklearn compatibility
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:  # Logits
                pred_classes = torch.argmax(predictions, dim=1)
            else:  # Already class indices
                pred_classes = predictions
            pred_classes = pred_classes.detach().cpu().numpy()
        else:
            pred_classes = predictions

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Store predictions and targets
        self.predictions.extend(pred_classes)
        self.targets.extend(targets)

        # Store probabilities if provided
        if probabilities is not None:
            if isinstance(probabilities, torch.Tensor):
                probabilities = probabilities.detach().cpu().numpy()
            self.probabilities.extend(probabilities)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            Dictionary containing all computed metrics
        """
        if not self.predictions:
            return {}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions,
                                               average='binary' if self.num_classes == 2 else 'macro')
        metrics['recall'] = recall_score(targets, predictions, average='binary' if self.num_classes == 2 else 'macro')
        metrics['f1'] = f1_score(targets, predictions, average='binary' if self.num_classes == 2 else 'macro')

        # Per-class metrics for binary classification
        if self.num_classes == 2:
            # Metrics for each class
            precision_per_class = precision_score(targets, predictions, average=None)
            recall_per_class = recall_score(targets, predictions, average=None)
            f1_per_class = f1_score(targets, predictions, average=None)

            metrics['precision_real'] = precision_per_class[0]
            metrics['precision_fake'] = precision_per_class[1]
            metrics['recall_real'] = recall_per_class[0]
            metrics['recall_fake'] = recall_per_class[1]
            metrics['f1_real'] = f1_per_class[0]
            metrics['f1_fake'] = f1_per_class[1]

            # Specificity and sensitivity (important for deepfake detection)
            cm = confusion_matrix(targets, predictions)
            tn, fp, fn, tp = cm.ravel()

            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # AUC-ROC if probabilities are available
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            if self.num_classes == 2:
                # Use probabilities for positive class (fake)
                pos_probs = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
                metrics['auc_roc'] = roc_auc_score(targets, pos_probs)
            else:
                # Multi-class AUC
                metrics['auc_roc'] = roc_auc_score(targets, probabilities, multi_class='ovr')

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for current predictions."""
        if not self.predictions:
            return np.array([])
        return confusion_matrix(self.targets, self.predictions)

    def plot_confusion_matrix(self, save_path: Optional[str] = None, class_names: List[str] = None):
        """
        Plot confusion matrix.

        Args:
            save_path: Path to save the plot
            class_names: Names of the classes
        """
        cm = self.get_confusion_matrix()
        if cm.size == 0:
            return

        if class_names is None:
            class_names = ['Real', 'Fake'] if self.num_classes == 2 else [f'Class_{i}' for i in range(self.num_classes)]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.

    Monitors validation metric and stops training when it stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy/F1
        restore_best_weights: Whether to restore best weights when stopping
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'max',
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None

        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:  # mode == 'max'
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score
            model: Model to potentially save weights from

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        # Check if we should stop
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"Restored best weights with score: {self.best_score:.4f}")
            return True

        return False


class MobileViTTrainer:
    """
    Complete training pipeline for MobileViT deepfake detection.

    This trainer includes all advanced features:
    - Mixed precision training (optimized for M4 Max)
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Comprehensive logging
    - Model checkpointing
    - Multiple optimization strategies

    Args:
        model: MobileViT model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 config: Dict[str, Any]):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup device (optimized for M4 Max)
        self.device = self._setup_device()
        self.model.to(self.device)

        # Setup mixed precision for M4 Max
        # Note: M4 Max uses different mixed precision than CUDA
        self.use_amp = config.get('use_mixed_precision', True)
        if self.use_amp and self.device.type == 'mps':
            # MPS doesn't use GradScaler like CUDA
            self.scaler = None
            print("Using MPS-optimized mixed precision")
        elif self.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
            print("Using CUDA mixed precision")
        else:
            self.scaler = None
            print("Mixed precision disabled")

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Setup loss function with class weights
        self.criterion = self._setup_loss_function()

        # Setup metrics
        self.metrics_calculator = MetricsCalculator(num_classes=config.get('num_classes', 2))

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.001),
            mode='max',  # We monitor F1 score
            restore_best_weights=True
        )

        # Setup logging
        self.logger = self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.training_history = defaultdict(list)

        print(f"Trainer initialized with device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _setup_device(self) -> torch.device:
        """Setup the best available device for M4 Max."""
        if torch.backends.mps.is_available():
            # M4 Max with Metal Performance Shaders
            device = torch.device('mps')
            print("Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA")
        else:
            device = torch.device('cpu')
            print("Using CPU")

        return device

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with advanced configurations."""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw')

        # Get model parameters, optionally with different learning rates for different parts
        params = []

        # Backbone parameters (lower learning rate)
        backbone_params = []
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                backbone_params.append(param)

        # Classifier parameters (higher learning rate)
        classifier_params = []
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)

        # Create parameter groups with different learning rates
        if backbone_params and classifier_params:
            base_lr = opt_config.get('lr', 1e-4)
            params = [
                {'params': backbone_params, 'lr': base_lr * 0.1},  # Lower LR for backbone
                {'params': classifier_params, 'lr': base_lr}  # Normal LR for classifier
            ]
        else:
            params = self.model.parameters()

        # Choose optimizer
        if opt_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 1e-2),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type.lower() == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=opt_config.get('lr', 1e-3),
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4),
                nesterov=opt_config.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        return optimizer

    def _setup_scheduler(self) -> None | CosineAnnealingLR | StepLR | ReduceLROnPlateau:
        """Setup learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        if not sched_config:
            return None

        sched_type = sched_config.get('type', 'cosine')

        if sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # We monitor F1 score
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5),
                min_lr=sched_config.get('min_lr', 1e-6)
            )
        else:
            return None

        return scheduler

    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function with class balancing."""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'cross_entropy')

        # Get class weights from training data if available
        class_weights = None
        if hasattr(self.train_loader.dataset, 'get_class_weights'):
            class_weights = self.train_loader.dataset.get_class_weights()
            class_weights = class_weights.to(self.device)
            print(f"Using class weights: {class_weights}")

        if loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == 'focal':
            # Focal loss for handling class imbalance
            criterion = FocalLoss(
                alpha=loss_config.get('alpha', 1.0),
                gamma=loss_config.get('gamma', 2.0),
                weight=class_weights
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return criterion

    def _setup_logging(self) -> SummaryWriter:
        """Setup TensorBoard logging."""
        log_dir = self.config.get('log_dir', f'runs/mobilevit_deepfake_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(log_dir, exist_ok=True)

        # Save config to log directory
        config_path = os.path.join(log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        return SummaryWriter(log_dir)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics for the epoch
        """
        self.model.train()
        self.metrics_calculator.reset()

        running_loss = 0.0
        num_batches = len(self.train_loader)

        # Progress tracking
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp and self.device.type == 'mps':
                # MPS mixed precision (different from CUDA)
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            elif self.use_amp and self.device.type == 'cuda':
                # CUDA mixed precision
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                # Regular precision
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass
            if self.scaler is not None:
                # CUDA mixed precision backward
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.get('gradient_clipping', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clipping']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular backward pass (MPS or CPU)
                loss.backward()

                # Gradient clipping
                if self.config.get('gradient_clipping', 0) > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clipping']
                    )

                self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                probabilities = F.softmax(outputs, dim=1)
                self.metrics_calculator.update(outputs, targets, probabilities)

            running_loss += loss.item()

            # Log batch progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                elapsed = time.time() - start_time
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, '
                      f'Loss: {loss.item():.4f}, Time: {elapsed:.2f}s')

        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.metrics_calculator.compute_metrics()
        epoch_metrics['loss'] = epoch_loss

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of validation metrics for the epoch
        """
        self.model.eval()
        self.metrics_calculator.reset()

        running_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move data to device
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                if self.use_amp and self.device.type == 'mps':
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                elif self.use_amp and self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                # Update metrics
                probabilities = F.softmax(outputs, dim=1)
                self.metrics_calculator.update(outputs, targets, probabilities)
                running_loss += loss.item()

        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.metrics_calculator.compute_metrics()
        epoch_metrics['loss'] = epoch_loss

        return epoch_metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config,
            'training_history': dict(self.training_history)
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved with F1: {self.best_f1:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_f1 = checkpoint['best_f1']
        self.training_history = defaultdict(list, checkpoint.get('training_history', {}))

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from epoch {self.current_epoch}")

    def train(self) -> Dict[str, List[float]]:
        """
        Complete training loop.

        Returns:
            Training history dictionary
        """
        num_epochs = self.config.get('epochs', 100)

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate_epoch()

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time

            # Console logging
            print(f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics.get('auc_roc', 0.0):.4f}")
            print(f"LR: {current_lr:.6f}")
            print("-" * 60)

            # TensorBoard logging
            for metric_name, metric_value in train_metrics.items():
                self.logger.add_scalar(f'Train/{metric_name}', metric_value, epoch)

            for metric_name, metric_value in val_metrics.items():
                self.logger.add_scalar(f'Validation/{metric_name}', metric_value, epoch)

            self.logger.add_scalar('Learning_Rate', current_lr, epoch)

            # Update training history
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)

            # Check for best model
            current_f1 = val_metrics['f1']
            is_best = current_f1 > self.best_f1
            if is_best:
                self.best_f1 = current_f1

            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(is_best)

            # Early stopping check
            if self.early_stopping(current_f1, self.model):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Final model save
        self.save_checkpoint(is_best=False)

        # Close logger
        self.logger.close()

        print("Training completed!")
        print(f"Best validation F1: {self.best_f1:.4f}")

        return dict(self.training_history)

    def evaluate(self, test_loader, save_results: bool = True) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader
            save_results: Whether to save detailed results

        Returns:
            Dictionary of test metrics
        """
        print("Evaluating on test set...")

        self.model.eval()
        self.metrics_calculator.reset()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                if self.use_amp and self.device.type == 'mps':
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        outputs = self.model(images)
                elif self.use_amp and self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)

                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # Update metrics
                self.metrics_calculator.update(outputs, targets, probabilities)

        # Compute final metrics
        test_metrics = self.metrics_calculator.compute_metrics()

        # Print detailed results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        for metric_name, metric_value in test_metrics.items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")

        # Save detailed results if requested
        if save_results:
            results_dir = self.config.get('results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)

            # Save confusion matrix plot
            cm_path = os.path.join(results_dir, 'confusion_matrix.png')
            self.metrics_calculator.plot_confusion_matrix(cm_path)

            # Save classification report
            report = classification_report(
                all_targets, all_predictions,
                target_names=['Real', 'Fake'],
                output_dict=True
            )

            report_path = os.path.join(results_dir, 'classification_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            # Save test metrics
            metrics_path = os.path.join(results_dir, 'test_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)

            print(f"Results saved to {results_dir}")

        return test_metrics


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    This loss function is particularly useful for deepfake detection where
    there might be class imbalance between real and fake samples.

    Args:
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        weight: Class weights
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy plot
    axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score plot
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # AUC plot (if available)
    if 'val_auc_roc' in history:
        axes[1, 1].plot(history['val_auc_roc'], label='Validation AUC-ROC')
        axes[1, 1].set_title('Validation AUC-ROC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC-ROC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test the training utilities
    print("Testing training utilities...")

    # Test metrics calculator
    metrics_calc = MetricsCalculator(num_classes=2)

    # Dummy predictions and targets
    dummy_preds = torch.tensor([0, 1, 1, 0, 1])
    dummy_targets = torch.tensor([0, 1, 0, 0, 1])
    dummy_probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]])

    metrics_calc.update(dummy_preds, dummy_targets, dummy_probs)
    metrics = metrics_calc.compute_metrics()

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTraining utilities test completed!")