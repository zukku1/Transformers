"""
Main Training Script for MobileViT Deepfake Detection

This is the main entry point for training the MobileViT model for deepfake detection.
It ties together all components: model, data, training pipeline, and evaluation.

Usage:
    python main.py --config configs/config.yaml
    python main.py --data_dir /path/to/dataset --epochs 100
    python main.py --resume checkpoints/best_model.pth

The script supports:
- Configuration via YAML files or command line arguments
- Resume training from checkpoints
- Automatic mixed precision for M4 Max
- Comprehensive logging and evaluation
- Easy hyperparameter experimentation
"""

import argparse
import os
import warnings

import torch
import torch.nn as nn
import yaml

warnings.filterwarnings('ignore')

# Import our custom modules
from models.mobilevit import mobilevit_s
from data.dataset import create_data_loaders
from training.trainer import MobileViTTrainer, plot_training_history
from utils.utils import print_model_summary
from utils.utils import setup_device_and_memory
from utils.utils import setup_logging

import sys


def parse_arguments():
    """
    Parse command line arguments for the training script.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Train MobileViT for Deepfake Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test subdirectories')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Training data directory (overrides data_dir/train)')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation data directory (overrides data_dir/val)')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Test data directory (overrides data_dir/test)')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='mobilevit_s',
                        choices=['mobilevit_s'], help='Model architecture to use')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (2 for binary deepfake detection)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pretrained weights')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay for optimizer')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')

    # Training optimizations
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    # Data augmentation
    parser.add_argument('--augmentation', type=str, default='advanced',
                        choices=['basic', 'advanced', 'face_aware'],
                        help='Data augmentation strategy')

    # Logging and checkpointing
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for results and plots')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Evaluation arguments
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate model without training')
    parser.add_argument('--test_time_augmentation', action='store_true',
                        help='Use test time augmentation for evaluation')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML config with command line arguments.
    Command line arguments take precedence.

    Args:
        config: Configuration from YAML file
        args: Command line arguments

    Returns:
        Merged configuration dictionary
    """
    # Convert args to dict and filter out None values
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    # Merge with config (args take precedence)
    merged_config = {**config, **args_dict}

    return merged_config


def setup_directories(config: dict):
    """
    Create necessary directories for logging, checkpoints, and results.

    Args:
        config: Configuration dictionary
    """
    directories = ['log_dir', 'checkpoint_dir', 'results_dir']

    for dir_key in directories:
        if dir_key in config:
            os.makedirs(config[dir_key], exist_ok=True)
            print(f"Created directory: {config[dir_key]}")


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducible results.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"Random seed set to: {seed}")


def create_model(config: dict) -> nn.Module:
    """
    Create and initialize the model.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized model
    """
    model_name = config.get('model_name', 'mobilevit_s')
    num_classes = config.get('num_classes', 2)
    pretrained = config.get('pretrained', False)

    print(f"Creating {model_name} model...")

    if model_name == 'mobilevit_s':
        model = mobilevit_s(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Print model summary
    input_size = config.get('image_size', 224)
    print_model_summary(model, input_size=(3, input_size, input_size))

    return model


def prepare_data_loaders(config: dict) -> tuple:
    """
    Prepare data loaders for training, validation, and testing.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Determine data directories
    data_dir = config['data_dir']
    train_dir = config.get('train_dir', os.path.join(data_dir, 'train'))
    val_dir = config.get('val_dir', os.path.join(data_dir, 'val'))
    test_dir = config.get('test_dir', os.path.join(data_dir, 'test'))

    # Validate directories exist
    for directory, name in [(train_dir, 'train'), (val_dir, 'validation')]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"{name.capitalize()} directory not found: {directory}")

    # Test directory is optional
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}. Skipping test data loading.")
        test_dir = None

    print(f"Loading data from:")
    print(f"  Train: {train_dir}")
    print(f"  Validation: {val_dir}")
    if test_dir:
        print(f"  Test: {test_dir}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=config.get('batch_size', 32),
        image_size=config.get('image_size', 224),
        num_workers=config.get('num_workers', 4),
        augmentation_type=config.get('augmentation', 'advanced')
    )

    return train_loader, val_loader, test_loader


def main():
    """
    Main training function that orchestrates the entire pipeline.

    This function:
    1. Parses arguments and loads configuration
    2. Sets up the environment (device, directories, logging)
    3. Creates model and data loaders
    4. Trains the model or evaluates if specified
    5. Saves results and generates plots
    """

    # Parse command line arguments
    global test_metrics, trainer
    args = parse_arguments()

    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        config = merge_config_with_args(config, args)
    else:
        # Use arguments as config
        config = vars(args)

    # Set experiment name if not provided
    if not config.get('experiment_name'):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment_name'] = f"mobilevit_deepfake_{timestamp}"

    # Update paths with experiment name
    base_log_dir = config.get('log_dir', 'runs')
    base_checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    base_results_dir = config.get('results_dir', 'results')

    config['log_dir'] = os.path.join(base_log_dir, config['experiment_name'])
    config['checkpoint_dir'] = os.path.join(base_checkpoint_dir, config['experiment_name'])
    config['results_dir'] = os.path.join(base_results_dir, config['experiment_name'])

    # Setup directories
    setup_directories(config)

    # Set random seeds for reproducibility
    set_random_seeds(config.get('seed', 42))

    # Setup device and memory optimizations for M4 Max
    device = setup_device_and_memory()
    print(f"Using device: {device}")

    # Setup logging
    logger = setup_logging(config['log_dir'])
    logger.info(f"Starting experiment: {config['experiment_name']}")
    logger.info(f"Configuration: {config}")

    try:
        # Create model
        model = create_model(config)

        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(config)

        # Handle evaluation-only mode
        if config.get('evaluate_only', False):
            if not config.get('resume'):
                raise ValueError("Must provide --resume checkpoint for evaluation-only mode")

            print("Running evaluation-only mode...")

            # Create trainer (needed for evaluation)
            trainer = MobileViTTrainer(model, train_loader, val_loader, config)

            # Load checkpoint
            trainer.load_checkpoint(config['resume'])

            # Evaluate on test set if available
            if test_loader:
                test_metrics = trainer.evaluate(test_loader, save_results=True)
                logger.info(f"Test metrics: {test_metrics}")
            else:
                print("No test data available for evaluation")

            return

        # Create trainer
        print("Initializing trainer...")
        trainer = MobileViTTrainer(model, train_loader, val_loader, config)

        # Resume from checkpoint if specified
        if config.get('resume'):
            print(f"Resuming training from: {config['resume']}")
            trainer.load_checkpoint(config['resume'])

        # Start training
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        training_history = trainer.train()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        # Plot training history
        history_plot_path = os.path.join(config['results_dir'], 'training_history.png')
        plot_training_history(training_history, save_path=history_plot_path)
        print(f"Training history plot saved to: {history_plot_path}")

        # Evaluate on test set if available
        if test_loader:
            print("\n" + "=" * 60)
            print("EVALUATING ON TEST SET")
            print("=" * 60)

            test_metrics = trainer.evaluate(test_loader, save_results=True)
            logger.info(f"Final test metrics: {test_metrics}")

            # Save final results summary
            summary = {
                'experiment_name': config['experiment_name'],
                'best_validation_f1': trainer.best_f1,
                'test_metrics': test_metrics,
                'config': config
            }

            summary_path = os.path.join(config['results_dir'], 'experiment_summary.json')
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"Experiment summary saved to: {summary_path}")

        # Log completion
        logger.info("Experiment completed successfully")
        logger.info(f"Best validation F1: {trainer.best_f1:.4f}")
        if test_loader:
            logger.info(f"Test F1: {test_metrics.get('f1', 'N/A'):.4f}")

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {config['results_dir']}")
        print(f"Best validation F1: {trainer.best_f1:.4f}")
        if test_loader and 'f1' in test_metrics:
            print(f"Test F1: {test_metrics['f1']:.4f}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        logger.info("Training interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        logger.error(f"Error during training: {e}")
        raise
    finally:
        # Clean up
        if 'trainer' in locals():
            trainer.logger.close()


def create_sample_config():
    """
    Create a sample configuration file for reference.

    This function generates a comprehensive YAML configuration file
    that demonstrates all available options for training.
    """
    sample_config = {
        # Data configuration
        'data_dir': '/path/to/your/deepfake/dataset',
        'batch_size': 32,
        'image_size': 224,
        'augmentation': 'advanced',  # 'basic', 'advanced', 'face_aware'

        # Model configuration
        'model_name': 'mobilevit_s',
        'num_classes': 2,
        'pretrained': True,

        # Training configuration
        'epochs': 100,
        'patience': 15,  # Early stopping patience
        'gradient_clipping': 1.0,
        'use_mixed_precision': True,

        # Optimizer configuration
        'optimizer': {
            'type': 'adamw',  # 'adamw' or 'sgd'
            'lr': 1e-4,
            'weight_decay': 1e-2,
            'betas': [0.9, 0.999]
        },

        # Scheduler configuration
        'scheduler': {
            'type': 'cosine',  # 'cosine', 'step', 'reduce_on_plateau'
            'eta_min': 1e-6,
            'step_size': 30,  # For step scheduler
            'gamma': 0.1,  # For step scheduler
            'factor': 0.5,  # For reduce_on_plateau
            'patience': 5  # For reduce_on_plateau
        },

        # Loss configuration
        'loss': {
            'type': 'cross_entropy',  # 'cross_entropy' or 'focal'
            'alpha': 1.0,  # For focal loss
            'gamma': 2.0  # For focal loss
        },

        # Logging configuration
        'log_interval': 100,  # Log every N batches
        'save_interval': 10,  # Save checkpoint every N epochs

        # Paths (will be auto-generated if not specified)
        'experiment_name': None,  # Auto-generated with timestamp
        'log_dir': 'runs',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',

        # System configuration
        'seed': 42,
        'num_workers': 4
    }

    config_path = 'configs/sample_config.yaml'
    os.makedirs('configs', exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)

    print(f"Sample configuration saved to: {config_path}")
    print("Edit this file and use it with: python main.py --config configs/sample_config.yaml")


if __name__ == "__main__":
    # Check if user wants to create sample config
    if len(sys.argv) > 1 and sys.argv[1] == '--create-sample-config':
        create_sample_config()
        sys.exit(0)

    # Show usage information if no arguments provided
    if len(sys.argv) == 1:
        print("MobileViT Deepfake Detection Training")
        print("=" * 50)
        print()
        print("Usage examples:")
        print("  # Create sample configuration file:")
        print("  python main.py --create-sample-config")
        print()
        print("  # Train with configuration file:")
        print("  python main.py --config configs/config.yaml")
        print()
        print("  # Train with command line arguments:")
        print("  python main.py --data_dir /path/to/dataset --epochs 100 --batch_size 32")
        print()
        print("  # Resume training from checkpoint:")
        print("  python main.py --config configs/config.yaml --resume checkpoints/best_model.pth")
        print()
        print("  # Evaluate only (no training):")
        print("  python main.py --config configs/config.yaml --resume checkpoints/best_model.pth --evaluate_only")
        print()
        print("For detailed help, use: python main.py --help")
        sys.exit(0)

    # Run main training
    main()

