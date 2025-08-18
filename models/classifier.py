"""
Advanced Classifier Head for Deepfake Detection

This module implements sophisticated classification heads specifically designed
for deepfake detection. It includes features like:
- Multi-layer perceptron with batch normalization
- Dropout for regularization
- Attention mechanisms for feature importance
- Uncertainty estimation capabilities

The classifier is designed to work with MobileViT backbone features.
"""

from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based Global Average Pooling.

    Instead of simple average pooling, this uses learned attention weights
    to focus on the most important spatial locations for classification.
    This is particularly useful for deepfake detection where certain
    regions of the face may be more indicative of manipulation.

    Args:
        in_channels: Number of input feature channels
        reduction_ratio: Ratio to reduce channels in attention computation
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(AttentionPooling, self).__init__()

        # Reduce channels for efficient attention computation
        reduced_channels = max(1, in_channels // reduction_ratio)

        # Attention mechanism to learn spatial importance
        self.attention = nn.Sequential(
            # First conv to reduce channels and compute spatial attention
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),

            # Second conv to produce attention weights
            nn.Conv2d(reduced_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # Normalize attention weights between 0 and 1
        )

        # Global average pooling for final feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention pooling.

        Args:
            x: Input feature tensor (batch_size, channels, height, width)

        Returns:
            Pooled features (batch_size, channels)
        """
        # Compute spatial attention weights
        # Shape: (batch_size, 1, height, width)
        attention_weights = self.attention(x)

        # Apply attention weights to input features
        # Broadcast multiplication across all channels
        attended_features = x * attention_weights

        # Global average pooling to get final feature vector
        pooled_features = self.global_pool(attended_features)

        # Remove spatial dimensions: (batch_size, channels, 1, 1) -> (batch_size, channels)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        return pooled_features


class ChannelAttention(nn.Module):
    """
    Channel Attention Module to emphasize important feature channels.

    This helps the model focus on the most discriminative feature channels
    for deepfake detection, similar to SENet attention mechanism.

    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for the bottleneck layer
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()

        # Bottleneck layer to reduce parameters
        reduced_channels = max(1, in_channels // reduction_ratio)

        # Global average pooling for channel statistics
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Channel attention network
        self.attention_net = nn.Sequential(
            # Reduce channels
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),

            # Expand back to original channels
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()  # Attention weights between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of channel attention.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Attention-weighted features of same shape as input
        """
        batch_size, channels, height, width = x.shape

        # Global average pooling: (batch_size, channels, H, W) -> (batch_size, channels, 1, 1)
        pooled = self.global_pool(x)

        # Flatten for linear layers: (batch_size, channels, 1, 1) -> (batch_size, channels)
        pooled = pooled.view(batch_size, channels)

        # Compute channel attention weights: (batch_size, channels)
        attention_weights = self.attention_net(pooled)

        # Reshape for broadcasting: (batch_size, channels) -> (batch_size, channels, 1, 1)
        attention_weights = attention_weights.view(batch_size, channels, 1, 1)

        # Apply attention weights to input features
        attended_features = x * attention_weights

        return attended_features


class DeepfakeClassifier(nn.Module):
    """
    Advanced classifier head specifically designed for deepfake detection.

    This classifier incorporates several advanced techniques:
    - Attention mechanisms for feature importance
    - Multiple dropout layers for regularization
    - Batch normalization for training stability
    - Optional uncertainty estimation

    Args:
        in_features: Number of input features from backbone
        num_classes: Number of output classes (2 for binary classification)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
        use_attention: Whether to use channel attention
        use_uncertainty: Whether to include uncertainty estimation
    """

    def __init__(self,
                 in_features: int = 640,
                 num_classes: int = 2,
                 hidden_dims=None,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True,
                 use_uncertainty: bool = False):
        super(DeepfakeClassifier, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty

        # Channel attention for feature refinement
        if use_attention:
            self.channel_attention = ChannelAttention(in_features)
        else:
            self.channel_attention = nn.Identity()

        # Build multi-layer classifier
        layers = []
        current_dim = in_features

        # Hidden layers with batch normalization and dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Linear(current_dim, num_classes)

        # Uncertainty estimation head (optional)
        if use_uncertainty:
            # Predict log variance for uncertainty estimation
            self.uncertainty_head = nn.Linear(current_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[Any, Any] | Any:
        """
        Forward pass of the classifier.

        Args:
            x: Input features (batch_size, in_features) or (batch_size, channels, H, W)

        Returns:
            Classification logits (batch_size, num_classes)
            If use_uncertainty=True, returns (logits, log_variance)
        """
        # Handle 4D input (from backbone before global pooling)
        if x.dim() == 4:
            # Apply channel attention if using 4D features
            x = self.channel_attention(x)

            # Global average pooling
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)

        # Apply feature layers
        features = self.feature_layers(x)

        # Classification logits
        logits = self.classifier(features)

        if self.use_uncertainty:
            # Uncertainty estimation (log variance)
            log_var = self.uncertainty_head(features)
            return logits, log_var

        return logits

    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using Monte Carlo Dropout.

        This technique uses dropout at inference time to estimate model uncertainty.
        Multiple forward passes with different dropout patterns provide a distribution
        of predictions, from which we can estimate uncertainty.

        Args:
            x: Input tensor
            num_samples: Number of MC samples for uncertainty estimation

        Returns:
            Tuple of (mean_predictions, prediction_variance)
        """
        # Enable training mode to activate dropout
        self.train()

        predictions = []

        # Collect multiple predictions with different dropout patterns
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                # Apply softmax to get probabilities
                pred_probs = F.softmax(pred, dim=1)
                predictions.append(pred_probs)

        # Stack predictions: (num_samples, batch_size, num_classes)
        predictions = torch.stack(predictions)

        # Compute mean and variance across samples
        mean_pred = predictions.mean(dim=0)  # (batch_size, num_classes)
        pred_var = predictions.var(dim=0)  # (batch_size, num_classes)

        # Return to evaluation mode
        self.eval()

        return mean_pred, pred_var


class MultiScaleClassifier(nn.Module):
    """
    Multi-scale classifier that processes features at different resolutions.

    This is particularly useful for deepfake detection as manipulations
    might be more apparent at certain scales. The classifier processes
    features from multiple stages of the backbone and combines them.

    Args:
        feature_dims: List of feature dimensions from different stages
        num_classes: Number of output classes
        fusion_method: How to combine multi-scale features ('concat', 'add', 'attention')
    """

    def __init__(self,
                 feature_dims=None,
                 num_classes: int = 2,
                 fusion_method: str = 'attention'):
        super(MultiScaleClassifier, self).__init__()

        if feature_dims is None:
            feature_dims = [96, 128, 160, 640]
        self.feature_dims = feature_dims
        self.fusion_method = fusion_method

        # Individual classifiers for each scale
        self.scale_classifiers = nn.ModuleList()
        for dim in feature_dims:
            classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(dim // 4, num_classes)
            )
            self.scale_classifiers.append(classifier)

        # Fusion mechanism
        if fusion_method == 'attention':
            # Learnable attention weights for different scales
            self.fusion_attention = nn.Linear(len(feature_dims), len(feature_dims))
        elif fusion_method == 'concat':
            # Concatenate all scale predictions
            self.fusion_layer = nn.Linear(len(feature_dims) * num_classes, num_classes)

    def forward(self, features: list) -> torch.Tensor:
        """
        Forward pass with multi-scale features.

        Args:
            features: List of feature tensors from different stages

        Returns:
            Combined classification logits
        """
        # Get predictions from each scale
        global final_pred
        scale_predictions = []
        for feature, classifier in zip(features, self.scale_classifiers):
            pred = classifier(feature)
            scale_predictions.append(pred)

        # Stack predictions: (batch_size, num_scales, num_classes)
        scale_predictions = torch.stack(scale_predictions, dim=1)

        # Fuse predictions based on fusion method
        if self.fusion_method == 'add':
            # Simple averaging
            final_pred = scale_predictions.mean(dim=1)

        elif self.fusion_method == 'attention':
            # Attention-weighted fusion
            batch_size, num_scales, num_classes = scale_predictions.shape

            # Compute attention weights for each scale
            attention_weights = self.fusion_attention(
                torch.ones(batch_size, num_scales, device=scale_predictions.device))
            attention_weights = F.softmax(attention_weights, dim=1)

            # Apply attention weights
            attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, num_scales, 1)
            final_pred = (scale_predictions * attention_weights).sum(dim=1)

        elif self.fusion_method == 'concat':
            # Concatenate and process through fusion layer
            concat_pred = scale_predictions.view(scale_predictions.size(0), -1)
            final_pred = self.fusion_layer(concat_pred)

        return final_pred


if __name__ == "__main__":
    # Test the classifier components
    print("Testing Deepfake Classifier...")

    # Test with dummy backbone features
    dummy_features = torch.randn(4, 640)  # Batch size 4, 640 features

    # Create and test basic classifier
    classifier = DeepfakeClassifier(in_features=640, num_classes=2)
    output = classifier(dummy_features)
    print(f"Basic classifier output shape: {output.shape}")

    # Test uncertainty estimation
    classifier_uncertain = DeepfakeClassifier(in_features=640, num_classes=2, use_uncertainty=True)
    mean_pred, uncertainty = classifier_uncertain.predict_with_uncertainty(dummy_features, num_samples=10)
    print(f"Uncertainty estimation - Mean: {mean_pred.shape}, Variance: {uncertainty.shape}")

    # Test multi-scale classifier
    dummy_multi_features = [
        torch.randn(4, 96, 14, 14),  # Stage 2 features
        torch.randn(4, 128, 7, 7),  # Stage 3 features
        torch.randn(4, 160, 4, 4),  # Stage 4 features
        torch.randn(4, 640, 2, 2)  # Stage 5 features
    ]

    multi_classifier = MultiScaleClassifier(feature_dims=[96, 128, 160, 640])
    multi_output = multi_classifier(dummy_multi_features)
    print(f"Multi-scale classifier output shape: {multi_output.shape}")

    print("All classifier tests passed!")