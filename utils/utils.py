"""
Utility Modules for MobileViT Deepfake Detection

This file contains various utility functions for:
1. Model summary and parameter counting
2. Device setup and memory optimization for M4 Max
3. Logging utilities with TensorBoard integration
4. Visualization and debugging tools

All utilities are optimized for Apple Silicon (M4 Max) environment.
"""

import logging
import os
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn


# ===== MODEL SUMMARY UTILITIES =====

def print_model_summary(model: nn.Module,
                        input_size: Tuple[int, int, int] = (3, 224, 224),
                        device: str = 'auto') -> Dict[str, Any]:
    """
    Print comprehensive model summary including parameters, memory usage, and layer details.

    This function provides detailed information about the model architecture,
    which is crucial for understanding the complexity and memory requirements.

    Args:
        model: PyTorch model to analyze
        input_size: Input tensor size (C, H, W)
        device: Device to run the analysis on ('auto', 'cpu', 'mps', 'cuda')

    Returns:
        Dictionary containing summary statistics
    """

    # Auto-detect device if not specified
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    device = torch.device(device)

    # Move model to device
    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)

    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    # Basic model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Device: {device}")
    print("-" * 80)

    # Parameter statistics
    print("PARAMETER STATISTICS:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"  Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    print("-" * 80)

    # Layer-wise summary
    print("LAYER-WISE SUMMARY:")
    print(f"{'Layer Name':<40} {'Output Shape':<20} {'Params':<15} {'Memory (MB)':<12}")
    print("-" * 87)

    total_memory = 0
    layer_count = 0

    def register_hook(module, name):
        """Register forward hook to capture layer information."""

        def hook(module, input, output):
            nonlocal total_memory, layer_count

            # Calculate parameters
            params = sum(p.numel() for p in module.parameters())

            # Calculate memory usage (approximate)
            if isinstance(output, torch.Tensor):
                output_memory = output.numel() * 4 / 1024 / 1024  # MB
                output_shape = str(list(output.shape))
            elif isinstance(output, (list, tuple)):
                output_memory = sum(o.numel() * 4 / 1024 / 1024 for o in output if isinstance(o, torch.Tensor))
                output_shape = f"Multiple ({len(output)})"
            else:
                output_memory = 0
                output_shape = "Unknown"

            total_memory += output_memory
            layer_count += 1

            # Print layer information
            layer_name = name if len(name) <= 39 else name[:36] + "..."
            print(f"{layer_name:<40} {output_shape:<20} {params:<15,} {output_memory:<12.2f}")

        return hook

    # Register hooks for all modules
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = register_hook(module, name)
            hooks.append(module.register_forward_hook(hook))

    # Forward pass to trigger hooks
    try:
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print("-" * 87)
    print(f"Total layers: {layer_count}")
    print(f"Total memory usage: {total_memory:.2f} MB")

    # Model complexity metrics
    print("-" * 80)
    print("COMPLEXITY METRICS:")

    # FLOPs estimation (approximate)
    def estimate_flops(model, input_size):
        """Rough FLOP estimation for the model."""
        total_flops = 0

        def flop_hook(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                # Conv2D FLOPs: output_elements * (kernel_size^2 * input_channels + bias)
                if isinstance(output, torch.Tensor):
                    output_elements = output.numel()
                    kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                    total_flops += output_elements * kernel_flops
            elif isinstance(module, nn.Linear):
                # Linear FLOPs: input_features * output_features
                total_flops += module.in_features * module.out_features

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(flop_hook))

        try:
            with torch.no_grad():
                _ = model(torch.randn(1, *input_size).to(device))
        except:
            pass

        for hook in hooks:
            hook.remove()

        return total_flops

    estimated_flops = estimate_flops(model, input_size)
    print(f"  Estimated FLOPs: {estimated_flops:,}")
    print(f"  FLOPs (GFLOPs): {estimated_flops / 1e9:.2f}")

    # Memory efficiency metrics
    params_memory = total_params * 4 / 1024 / 1024  # Model parameters memory
    activation_memory = total_memory  # Activation memory
    total_model_memory = params_memory + activation_memory

    print(f"  Parameter memory: {params_memory:.2f} MB")
    print(f"  Activation memory: {activation_memory:.2f} MB")
    print(f"  Total model memory: {total_model_memory:.2f} MB")

    print("=" * 80)

    # Return summary statistics
    summary_stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024,
        'estimated_flops': estimated_flops,
        'estimated_gflops': estimated_flops / 1e9,
        'param_memory_mb': params_memory,
        'activation_memory_mb': activation_memory,
        'total_memory_mb': total_model_memory,
        'layer_count': layer_count
    }

    return summary_stats


def compare_models(*models, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """
    Compare multiple models side by side.

    Args:
        *models: Variable number of models to compare
        input_size: Input tensor size for comparison
    """
    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    print(f"{'Model':<30} {'Params':<15} {'Size (MB)':<12} {'GFLOPs':<10} {'Memory (MB)':<12}")
    print("-" * 100)

    for i, model in enumerate(models):
        stats = print_model_summary(model, input_size, device='cpu')
        model_name = f"Model {i + 1}" if not hasattr(model, '__class__') else model.__class__.__name__

        print(f"{model_name:<30} {stats['total_params']:<15,} {stats['model_size_mb']:<12.2f} "
              f"{stats['estimated_gflops']:<10.2f} {stats['total_memory_mb']:<12.2f}")

    print("=" * 100)


# ===== DEVICE AND MEMORY UTILITIES =====

def setup_device_and_memory() -> torch.device:
    """
    Setup optimal device and memory configurations for M4 Max.

    This function configures PyTorch for optimal performance on Apple Silicon,
    with fallbacks for other platforms.

    Returns:
        Configured device object
    """
    print("Setting up device and memory configurations...")

    # Detect available devices
    available_devices = []

    if torch.backends.mps.is_available():
        available_devices.append('mps')
    if torch.cuda.is_available():
        available_devices.append('cuda')
    available_devices.append('cpu')

    print(f"Available devices: {available_devices}")

    # Choose best device
    if 'mps' in available_devices:
        device = torch.device('mps')
        print("Using Apple Silicon MPS (Metal Performance Shaders)")

        # MPS-specific optimizations
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory caching

        # Check MPS capabilities
        print("MPS Device Info:")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  MPS built: {torch.backends.mps.is_built()}")

    elif 'cuda' in available_devices:
        device = torch.device('cuda')
        print("Using CUDA")

        # CUDA-specific optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

        # Print CUDA info
        print("CUDA Device Info:")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        if torch.cuda.is_available():
            print(f"  Device name: {torch.cuda.get_device_name()}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    else:
        device = torch.device('cpu')
        print("Using CPU")

        # CPU-specific optimizations
        torch.set_num_threads(min(8, torch.get_num_threads()))  # Limit CPU threads
        print(f"  CPU threads: {torch.get_num_threads()}")

    # System memory info
    memory_info = psutil.virtual_memory()
    print(f"System Memory:")
    print(f"  Total: {memory_info.total / 1024 ** 3:.2f} GB")
    print(f"  Available: {memory_info.available / 1024 ** 3:.2f} GB")
    print(f"  Used: {memory_info.percent:.1f}%")

    return device


def monitor_memory_usage(device: torch.device):
    """
    Monitor and print current memory usage.

    Args:
        device: Device to monitor
    """
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    elif device.type == 'mps':
        # MPS doesn't have direct memory monitoring, so we monitor system memory
        memory_info = psutil.virtual_memory()
        print(f"System Memory - Used: {memory_info.percent:.1f}% ({memory_info.used / 1024 ** 3:.2f} GB)")
    else:
        memory_info = psutil.virtual_memory()
        print(f"CPU Memory - Used: {memory_info.percent:.1f}% ({memory_info.used / 1024 ** 3:.2f} GB)")


def clear_memory_cache(device: torch.device):
    """
    Clear memory cache for the specified device.

    Args:
        device: Device to clear cache for
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("CUDA memory cache cleared")
    elif device.type == 'mps':
        torch.mps.empty_cache()
        print("MPS memory cache cleared")
    else:
        # For CPU, trigger garbage collection
        import gc
        gc.collect()
        print("Garbage collection triggered")


# ===== LOGGING UTILITIES =====

def setup_logging(log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup comprehensive logging with file and console handlers.

    Args:
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        Configured logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('mobilevit_deepfake')
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler for detailed logs
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # Error file handler
    error_file = os.path.join(log_dir, 'errors.log')
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)

    logger.info(f"Logging setup complete. Logs will be saved to: {log_file}")

    return logger


class PerformanceProfiler:
    """
    Performance profiler for training operations.

    This class helps identify bottlenecks in the training pipeline
    by measuring execution time of different components.
    """

    def __init__(self):
        self.timings = {}
        self.start_times = {}

    def start(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()

    def end(self, operation: str):
        """End timing an operation."""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(elapsed)
            del self.start_times[operation]

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary statistics."""
        summary = {}
        for operation, times in self.timings.items():
            summary[operation] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }
        return summary

    def print_summary(self):
        """Print timing summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE PROFILING SUMMARY")
        print("=" * 60)

        summary = self.get_summary()

        print(f"{'Operation':<20} {'Mean (s)':<10} {'Std (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Count':<8}")
        print("-" * 68)

        for operation, stats in summary.items():
            print(f"{operation:<20} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                  f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['count']:<8}")

        print("=" * 60)

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.start_times.clear()


# ===== VISUALIZATION UTILITIES =====

def plot_model_architecture(model: nn.Module, save_path: Optional[str] = None):
    """
    Create a visual representation of the model architecture.

    Args:
        model: Model to visualize
        save_path: Path to save the plot
    """
    try:
        from torchview import draw_graph

        # Create model graph
        model_graph = draw_graph(
            model,
            input_size=(1, 3, 224, 224),
            expand_nested=True,
            graph_name='MobileViT Architecture'
        )

        if save_path:
            model_graph.visual_graph.render(save_path, format='png', cleanup=True)
            print(f"Model architecture saved to: {save_path}")
        else:
            model_graph.visual_graph.view()

    except ImportError:
        print("torchview not available. Install with: pip install torchview")
        # Fallback: create simple text representation
        print("\nModel Architecture (Text Representation):")
        print("-" * 50)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                print(f"{name}: {module.__class__.__name__} (params: {params:,})")


def visualize_feature_maps(model: nn.Module,
                           input_tensor: torch.Tensor,
                           layer_name: str,
                           save_path: Optional[str] = None,
                           max_channels: int = 16):
    """
    Visualize feature maps from a specific layer.

    This is useful for understanding what the model learns at different stages.

    Args:
        model: Model to extract features from
        input_tensor: Input tensor to pass through the model
        layer_name: Name of the layer to visualize
        save_path: Path to save the visualization
        max_channels: Maximum number of channels to visualize
    """

    # Hook to capture feature maps
    feature_maps = {}

    def hook_fn(module, input, output):
        feature_maps[layer_name] = output.detach()

    # Register hook
    target_module = dict(model.named_modules())[layer_name]
    hook = target_module.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hook
    hook.remove()

    # Get feature maps
    if layer_name in feature_maps:
        features = feature_maps[layer_name][0]  # First batch item
        num_channels = min(features.shape[0], max_channels)

        # Create subplot grid
        cols = 4
        rows = (num_channels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_channels):
            row = i // cols
            col = i % cols

            # Normalize feature map for visualization
            feature_map = features[i].cpu().numpy()
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

            axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'Channel {i}')
            axes[row, col].axis('off')

        # Hide unused subplots
        for i in range(num_channels, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.suptitle(f'Feature Maps: {layer_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Feature maps saved to: {save_path}")
        else:
            plt.show()
    else:
        print(f"Layer {layer_name} not found in model")


def plot_attention_maps(model: nn.Module,
                        input_tensor: torch.Tensor,
                        save_path: Optional[str] = None):
    """
    Visualize attention maps from transformer blocks.

    Args:
        model: Model containing transformer blocks
        input_tensor: Input tensor
        save_path: Path to save the visualization
    """

    attention_maps = {}

    def attention_hook(module, input, output):
        # This would need to be adapted based on your specific attention implementation
        # For now, we'll create a placeholder
        if hasattr(module, 'attention_weights'):
            attention_maps[str(module)] = module.attention_weights.detach()

    # Register hooks on attention modules
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower():
            hooks.append(module.register_forward_hook(attention_hook))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Visualize attention maps if any were captured
    if attention_maps:
        fig, axes = plt.subplots(1, len(attention_maps), figsize=(5 * len(attention_maps), 5))
        if len(attention_maps) == 1:
            axes = [axes]

        for i, (module_name, attn_map) in enumerate(attention_maps.items()):
            # Average attention across heads and select first item in batch
            if attn_map.dim() == 4:  # (batch, heads, seq_len, seq_len)
                attn_map = attn_map[0].mean(0)  # Average across heads

            axes[i].imshow(attn_map.cpu().numpy(), cmap='viridis')
            axes[i].set_title(f'Attention: {module_name}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Attention maps saved to: {save_path}")
        else:
            plt.show()
    else:
        print("No attention maps captured. Check if model has attention modules.")


# ===== DEBUGGING UTILITIES =====

def debug_model_gradients(model: nn.Module):
    """
    Debug gradient flow in the model.

    This function helps identify vanishing or exploding gradient problems.

    Args:
        model: Model to debug
    """
    print("\n" + "=" * 60)
    print("GRADIENT ANALYSIS")
    print("=" * 60)

    total_norm = 0
    param_count = 0

    print(f"{'Layer Name':<40} {'Gradient Norm':<15} {'Param Shape':<20}")
    print("-" * 75)

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1

            layer_name = name if len(name) <= 39 else name[:36] + "..."
            print(f"{layer_name:<40} {param_norm:<15.6f} {str(list(param.shape)):<20}")
        else:
            layer_name = name if len(name) <= 39 else name[:36] + "..."
            print(f"{layer_name:<40} {'No gradient':<15} {str(list(param.shape)):<20}")

    total_norm = total_norm ** (1. / 2)

    print("-" * 75)
    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Parameters with gradients: {param_count}")

    # Gradient health check
    if total_norm > 100:
        print("⚠️  WARNING: Large gradient norm detected! Consider gradient clipping.")
    elif total_norm < 1e-6:
        print("⚠️  WARNING: Very small gradient norm detected! Check for vanishing gradients.")
    else:
        print("✅ Gradient norm looks healthy.")

    print("=" * 60)


def check_model_device_consistency(model: nn.Module, input_tensor: torch.Tensor):
    """
    Check if all model parameters and input are on the same device.

    Args:
        model: Model to check
        input_tensor: Input tensor to check
    """
    model_devices = {param.device for param in model.parameters()}
    input_device = input_tensor.device

    print(f"Input device: {input_device}")
    print(f"Model devices: {model_devices}")

    if len(model_devices) > 1:
        print("⚠️  WARNING: Model parameters are on different devices!")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.device}")
    elif input_device not in model_devices:
        print("⚠️  WARNING: Input and model are on different devices!")
    else:
        print("✅ Device consistency check passed.")


def profile_model_inference(model: nn.Module,
                            input_tensor: torch.Tensor,
                            num_runs: int = 100,
                            warmup_runs: int = 10):
    """
    Profile model inference speed.

    Args:
        model: Model to profile
        input_tensor: Input tensor for profiling
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    print(f"\nProfiling model inference speed...")
    print(f"Device: {device}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {num_runs}")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # Synchronize device if needed
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)

            # Synchronize device if needed
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\nInference Speed Results:")
    print(f"  Mean time: {mean_time * 1000:.2f} ms")
    print(f"  Std dev: {std_time * 1000:.2f} ms")
    print(f"  Min time: {min_time * 1000:.2f} ms")
    print(f"  Max time: {max_time * 1000:.2f} ms")
    print(f"  FPS: {1 / mean_time:.2f}")

    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': 1 / mean_time
    }


# ===== EXPORT UTILITIES =====

def save_model_for_deployment(model: nn.Module,
                              save_path: str,
                              input_size: Tuple[int, int, int] = (3, 224, 224),
                              optimize: bool = True):
    """
    Save model in optimized format for deployment.

    Args:
        model: Model to save
        save_path: Path to save the model
        input_size: Input tensor size for tracing
        optimize: Whether to optimize the model
    """
    model.eval()
    device = next(model.parameters()).device

    # Create dummy input for tracing
    dummy_input = torch.randn(1, *input_size).to(device)

    try:
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)

        if optimize:
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)

        # Save traced model
        traced_model.save(save_path)
        print(f"Optimized model saved to: {save_path}")

        # Test the saved model
        loaded_model = torch.jit.load(save_path)
        with torch.no_grad():
            original_output = model(dummy_input)
            traced_output = loaded_model(dummy_input)

            # Check if outputs are close
            if torch.allclose(original_output, traced_output, atol=1e-5):
                print("✅ Model tracing verification passed.")
            else:
                print("⚠️  WARNING: Model tracing verification failed!")

    except Exception as e:
        print(f"Error during model tracing: {e}")
        # Fallback: save state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__
        }, save_path.replace('.pt', '_state_dict.pt'))
        print(f"Fallback: Model state dict saved to: {save_path.replace('.pt', '_state_dict.pt')}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test with a simple model
    test_model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 2)
    )

    # Test model summary
    summary_stats = print_model_summary(test_model, (3, 224, 224))
    print(f"Summary stats: {summary_stats}")

    # Test device setup
    device = setup_device_and_memory()

    # Test performance profiler
    profiler = PerformanceProfiler()
    profiler.start('test_operation')
    time.sleep(0.01)  # Simulate work
    profiler.end('test_operation')
    profiler.print_summary()

    print("Utility functions test completed!")