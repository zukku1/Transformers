"""
MobileViT Architecture - Complete implementation of MobileViT-S for image classification

This file implements the full MobileViT-S architecture as described in the paper:
"MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"

The architecture combines the efficiency of MobileNets with the modeling power of
Vision Transformers, making it suitable for mobile deployment while maintaining
high accuracy.

Key components:
1. Stem: Initial feature extraction
2. MobileNet blocks: Efficient local feature extraction
3. MobileViT blocks: Global feature modeling with transformers
4. Classification head: Final prediction layers
"""

import torch
import torch.nn as nn

from .components import MobileViTBlock


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution - core building block of MobileNets.

    This replaces standard convolution with two separate operations:
    1. Depthwise convolution: applies a single filter per input channel
    2. Pointwise convolution: 1x1 conv to combine channels

    This dramatically reduces parameters and computations while maintaining
    similar representational power.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding for the convolution
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution: each input channel is convolved with its own filter
        # groups=in_channels means each input channel gets its own filter
        # This reduces parameters from (in_ch * out_ch * k * k) to (in_ch * k * k)
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,  # Key parameter for depthwise convolution
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        # Pointwise convolution: 1x1 conv to combine information across channels
        # This is where cross-channel communication happens
        # Parameters: (in_ch * out_ch * 1 * 1) = (in_ch * out_ch)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of depthwise separable convolution.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Apply depthwise convolution first
        x = self.depthwise(x)

        # Then apply pointwise convolution
        x = self.pointwise(x)

        return x



class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block - building block used in MobileNetV2 and MobileViT.

    Unlike traditional residual blocks that compress then expand, inverted residual
    blocks expand then compress, which works better for mobile architectures.

    Structure: 1x1 expand -> 3x3 depthwise -> 1x1 project (+ residual if applicable)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the depthwise convolution
        expand_ratio: How much to expand channels in the first layer
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 expand_ratio: int = 4):
        super(InvertedResidualBlock, self).__init__()

        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Calculate expanded channels
        expanded_channels = in_channels * expand_ratio

        # Build the block layers
        layers = []

        # 1. Expansion phase (only if expand_ratio > 1)
        if expand_ratio > 1:
            layers.extend([
                # 1x1 convolution to expand channels
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            ])

        # 2. Depthwise convolution phase
        layers.extend([
            # 3x3 depthwise convolution for spatial feature extraction
            nn.Conv2d(
                expanded_channels, expanded_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=expanded_channels,  # Depthwise convolution
                bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        ])

        # 3. Projection phase
        layers.extend([
            # 1x1 convolution to project back to output channels
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
            # Note: No activation after final projection (linear bottleneck)
        ])

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of inverted residual block.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection if applicable
        """
        # Apply the convolution block
        out = self.conv_block(x)

        # Add residual connection if dimensions match and stride is 1
        if self.use_residual:
            out = out + x

        return out



class MobileViT(nn.Module):
    """
    Complete MobileViT-S architecture for image classification.

    This implements the MobileViT-S variant which provides a good balance
    between accuracy and efficiency. The architecture progressively extracts
    features using a combination of MobileNet blocks and MobileViT blocks.

    Architecture stages:
    1. Stem: Initial feature extraction
    2. Early stages: MobileNet blocks for local features
    3. Middle stages: MobileViT blocks for global features
    4. Late stages: More MobileNet blocks
    5. Classification head: Global pooling + classifier

    Args:
        num_classes: Number of output classes
        image_size: Input image size (assumed square)
        in_channels: Number of input channels (3 for RGB)
    """

    def __init__(self,
                 num_classes: int = 2,  # Binary classification for deepfake detection
                 image_size: int = 224,
                 in_channels: int = 3):
        super(MobileViT, self).__init__()

        self.num_classes = num_classes
        self.image_size = image_size

        # Stem: Initial feature extraction
        # This converts RGB input to initial feature maps
        self.stem = nn.Sequential(
            # 3x3 convolution with stride 2 for initial downsampling
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU()
        )

        # Stage 1: Initial MobileNet blocks
        # These extract low-level local features
        self.stage1 = nn.Sequential(
            InvertedResidualBlock(16, 32, stride=1, expand_ratio=1),
            InvertedResidualBlock(32, 64, stride=2, expand_ratio=4),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=4)
        )

        # Stage 2: More MobileNet blocks with increased capacity
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(64, 96, stride=2, expand_ratio=4),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=4),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=4)
        )

        # Stage 3: First MobileViT block for global feature modeling
        # This is where we start using transformers for global context
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(96, 128, stride=2, expand_ratio=4),
            MobileViTBlock(
                in_channels=128,
                out_channels=128,
                patch_size=2,
                embed_dim=144,
                num_heads=4,
                num_layers=2,
                dropout=0.1
            )
        )

        # Stage 4: Second MobileViT block with more capacity
        # Deeper transformer processing for complex global patterns
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(128, 160, stride=2, expand_ratio=4),
            MobileViTBlock(
                in_channels=160,
                out_channels=160,
                patch_size=2,
                embed_dim=192,
                num_heads=4,
                num_layers=4,
                dropout=0.1
            )
        )

        # Stage 5: Final MobileViT block with highest capacity
        # Most sophisticated global feature modeling
        self.stage5 = nn.Sequential(
            InvertedResidualBlock(160, 640, stride=2, expand_ratio=4),
            MobileViTBlock(
                in_channels=640,
                out_channels=640,
                patch_size=2,
                embed_dim=240,
                num_heads=4,
                num_layers=3,
                dropout=0.1
            )
        )

        # Global average pooling to convert spatial features to vector
        # This removes spatial dimensions while preserving channel information
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head with multiple layers and dropout
        # This provides non-linear decision boundary for deepfake detection
        self.classifier = nn.Sequential(
            # First fully connected layer with dropout
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),

            # Second fully connected layer with dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),

            # Final classification layer
            nn.Linear(128, num_classes)
        )

        # Initialize weights for better training stability
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using appropriate strategies.

        This helps with training stability and convergence speed.
        Different layer types need different initialization strategies.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Use He initialization for conv layers with ReLU-like activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                # Initialize batch norm parameters
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                # Use He initialization for linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complete MobileViT network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Validate input dimensions
        batch_size = x.shape[0]

        # Stage-by-stage feature extraction
        # Each stage processes features at different spatial resolutions

        # Stem: Initial feature extraction
        # Input: (batch_size, 3, H, W) -> Output: (batch_size, 16, H/2, W/2)
        x = self.stem(x)

        # Stage 1: Local feature extraction with MobileNet blocks
        # Input: (batch_size, 16, H/2, W/2) -> Output: (batch_size, 64, H/4, W/4)
        x = self.stage1(x)

        # Stage 2: More sophisticated local features
        # Input: (batch_size, 64, H/4, W/4) -> Output: (batch_size, 96, H/8, W/8)
        x = self.stage2(x)

        # Stage 3: First global modeling with MobileViT
        # Input: (batch_size, 96, H/8, W/8) -> Output: (batch_size, 128, H/16, W/16)
        x = self.stage3(x)

        # Stage 4: Enhanced global modeling
        # Input: (batch_size, 128, H/16, W/16) -> Output: (batch_size, 160, H/32, W/32)
        x = self.stage4(x)

        # Stage 5: Final high-level feature extraction
        # Input: (batch_size, 160, H/32, W/32) -> Output: (batch_size, 640, H/64, W/64)
        x = self.stage5(x)

        # Global average pooling to remove spatial dimensions
        # Input: (batch_size, 640, H/64, W/64) -> Output: (batch_size, 640, 1, 1)
        x = self.global_pool(x)

        # Flatten for classification head
        # Input: (batch_size, 640, 1, 1) -> Output: (batch_size, 640)
        x = x.view(batch_size, -1)

        # Classification head for final prediction
        # Input: (batch_size, 640) -> Output: (batch_size, num_classes)
        logits = self.classifier(x)

        return logits

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Extract intermediate feature maps for visualization and analysis.

        This is useful for understanding what the model learns at different stages
        and for debugging purposes.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Dictionary containing feature maps from each stage
        """
        features = {}

        # Extract features from each stage
        x = self.stem(x)
        features['stem'] = x

        x = self.stage1(x)
        features['stage1'] = x

        x = self.stage2(x)
        features['stage2'] = x

        x = self.stage3(x)
        features['stage3'] = x

        x = self.stage4(x)
        features['stage4'] = x

        x = self.stage5(x)
        features['stage5'] = x

        return features



def mobilevit_s(num_classes: int = 2, pretrained: bool = False) -> MobileViT:
    """
    Create MobileViT-S model for deepfake detection.

    Args:
        num_classes: Number of output classes (2 for binary classification)
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        MobileViT model instance
    """
    model = MobileViT(num_classes=num_classes)

    if pretrained:
        # Load pretrained weights if available
        # Note: This would typically load ImageNet weights and adapt for deepfake detection
        try:
            # Placeholder for actual pretrained weight loading
            # In practice, you would download and load pretrained weights here
            print("Pretrained weights loading is not implemented yet.")
            print("Training from scratch...")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Training from scratch...")

    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count the number of parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


if __name__ == "__main__":
    # Test the model creation and forward pass
    print("Creating MobileViT-S model...")
    model = mobilevit_s(num_classes=2)

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Model test successful!")