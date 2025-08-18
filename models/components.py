"""
MobileViT Components - Individual building blocks of the MobileViT architecture

This file contains all the fundamental components that make up MobileViT:
1. Multi-Head Self-Attention (the core of transformers)
2. Feed-Forward Network (MLP in transformers)
3. Transformer Block (combines attention + FFN)
4. MobileViT Block (the key innovation combining CNNs and transformers)

Each component is explained line-by-line for educational purposes.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism - the core of transformer architecture.

    This allows the model to attend to different positions simultaneously,
    capturing both local and global dependencies in the input.

    Args:
        embed_dim: The embedding dimension (d_model in transformer papers) (1 dimensional vector representation of an image patch)
        num_heads: Number of attention heads to split the embedding into
        (every embedding is gone to every head and each head takes a part of each embedding depending on num_heads, head_dim = embed_dim // num_heads)
        dropout: Dropout probability for attention weights
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()

        # Store configuration parameters
        self.embed_dim = embed_dim  # Total embedding dimension
        self.num_heads = num_heads  # Number of parallel attention heads
        self.dropout = dropout

        # Each head will have dimension = embed_dim / num_heads
        # This ensures total parameters remain the same as single-head attention
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Scale factor for attention scores (prevents gradients from becoming too small)
        # This is sqrt(d_k) from the "Attention is All You Need" paper
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Query, Key, Value
        # These transform input embeddings into Q, K, V representations
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection to combine all heads back to original dimension
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)
            => x is a 3-dimensional tensor
            x consists of batch_size numbers of 2-dimensional tensors
            each 2-dimensional tensor represents an image
            each 2-dimensional tensor has num_patches numbers of 1-dimensional tensors
            each 1-dimensional tensor represents a patch
            each 1-dimensional tensor has embed_dim numbers of elements
            each element represents a feature (no. of features (embed_dim) is determined by programmer)

        Returns:
            Output tensor of same shape as input
        """
        batch_size, num_patches, embed_dim = x.shape

        # Step 1: Generate Query, Key, Value matrices
        # Each projection creates a linear transformation of the input
        # although inputs are same (x), nn.Linear function assigns random weights resulting in different outputs
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Step 2: Reshape for multi-head attention
        # We split embed_dim into num_heads of head_dim each
        # This allows parallel processing of different representation subspaces
        queries = queries.view(batch_size, num_patches, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_patches, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_patches, self.num_heads, self.head_dim)
        """
        now, queries, keys, and values are 4-dimensional tensors
        each 4-dimensional tensors consists of batch_size numbers of 3-dimensional tensors(images)
        each 3-dimensional tensor has num_patches numbers of 2-dimensional tensors(patches)
        each 2-dimensional tensor has num_heads numbers of 1-dimensional tensors(attention heads)
        each 1-dimensional tensor has head_dim numbers of elements (head_dim = embed_dim/num_heads)
        head_dim = number of features each attention head focuses on
        """

        # Transpose to bring heads dimension forward for batch processing
        # Shape becomes: (batch_size, num_heads, num_patches, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        """
        now, queries, keys, and values are 4-dimensional tensors
        each 4-dimensional tensors consists of batch_size numbers of 3-dimensional tensors(images)
        each 3-dimensional tensor has num_heads numbers of 2-dimensional tensors(attention heads)
        each 2-dimensional tensor has num_patches numbers of 1-dimensional tensors(patches)
        each 1-dimensional tensor has head_dim numbers of elements (head_dim = embed_dim/num_heads)
        """

        """
        # Compute Attention
        # Important Point: This measures how much each patch should attend to every other patch
        # Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        """

        # Step 3: Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))

        # Scale attention scores to prevent vanishing gradients
        attention_scores = attention_scores / self.scale

        # Step 4: Apply softmax to get attention weights
        # This normalizes scores so they sum to 1 across the sequence dimension
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights for regularization
        attention_weights = self.dropout_layer(attention_weights)

        # Step 5: Apply attention weights to values
        # This creates the final attention output by weighted combination of values
        attention_output = torch.matmul(attention_weights, values)

        # Step 6: Concatenate all heads back together
        # Transpose back to original layout: (batch_size, num_patches, num_heads, head_dim)
        attention_output = attention_output.transpose(1, 2)

        # Reshape to combine all heads: (batch_size, num_patches, embed_dim)
        attention_output = attention_output.contiguous().view(
            batch_size, num_patches, embed_dim
        )

        # Step 7: Final linear projection
        # This allows the model to learn how to combine information from all heads
        output = self.output_proj(attention_output)

        return output




class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network (FFN) - the MLP component of transformer blocks.

    This applies two linear transformations with a non-linear activation in between.
    It processes each position independently and identically.

    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension (usually 4x larger than embed_dim)
        dropout: Dropout probability for regularization
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForwardNetwork, self).__init__()

        # First linear transformation: expand to hidden dimension
        # This increases the model's capacity to learn complex patterns
        self.linear1 = nn.Linear(embed_dim, hidden_dim)

        # Non-linear activation function
        # SiLU (Swish) is used in MobileViT instead of ReLU for better performance
        self.activation = nn.SiLU()  # SiLU(x) = x * sigmoid(x)

        # Dropout for regularization after activation
        self.dropout = nn.Dropout(dropout)

        # Second linear transformation: project back to original dimension
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)

        Returns:
            Output tensor of same shape as input
        """
        # Expand to hidden dimension
        x = self.linear1(x)

        # Apply non-linear activation
        x = self.activation(x)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Project back to original dimension
        x = self.linear2(x)

        return x



class TransformerBlock(nn.Module):
    """
    Single Transformer Block - combines self-attention and feed-forward network.

    This is the fundamental building block of transformer architectures.
    It uses residual connections and layer normalization for stable training.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension for FFN
        dropout: Dropout probability
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()

        # Layer normalization before self-attention (Pre-LN transformer)
        # This normalizes inputs to have zero mean and unit variance
        # Pre-LN is more stable for training than Post-LN
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)

        # Multi-head self-attention layer
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # Layer normalization before feed-forward network
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # Feed-forward network
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim, dropout)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.

        The structure follows: x -> LN -> Attention -> Dropout -> Add -> LN -> FFN -> Dropout -> Add
        This is the Pre-LN transformer architecture.

        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)

        Returns:
            Output tensor of same shape as input
        """
        # First sub-layer: Self-Attention with residual connection
        # Apply layer norm first (Pre-LN architecture)
        normed_x = self.norm1(x)

        # Apply self-attention
        attention_output = self.attention(normed_x)

        # Apply dropout and add residual connection
        x = x + self.dropout(attention_output)

        # Second sub-layer: Feed-Forward Network with residual connection
        # Apply layer norm first
        normed_x = self.norm2(x)

        # Apply feed-forward network
        ffn_output = self.ffn(normed_x)

        # Apply dropout and add residual connection
        x = x + self.dropout(ffn_output)

        return x

class MobileViTBlock(nn.Module):
    """
    MobileViT Block - The key innovation that combines CNNs and Transformers.

    This block performs the following operations:
    1. Local feature extraction using depthwise convolutions
    2. Patch creation and linear projection
    3. Global feature modeling using transformers
    4. Patch reconstruction and feature fusion

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        patch_size: Size of patches for transformer processing
        embed_dim: Embedding dimension for transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 patch_size: int = 2,
                 embed_dim: int = 144,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super(MobileViTBlock, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Local representation using standard convolution
        # This extracts local features before global modeling
        self.local_conv = nn.Sequential(
            # 3x3 convolution for local feature extraction
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        # Linear projection to embed patches into transformer space
        # Each patch of size (patch_size x patch_size x in_channels) -> embed_dim
        # embed_dim is used to represent no. of features and its value is determined by programmer
        # embed_dim is actually used to reduce the value of patch_dim (a larger value) to a smaller value
        patch_dim = patch_size * patch_size * in_channels
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Stack of transformer blocks for global feature modeling
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=embed_dim * 4,  # Standard 4x expansion in FFN
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Project back from transformer embedding to spatial features
        self.patch_reconstruct = nn.Linear(embed_dim, patch_dim)

        # Final convolution to adjust output channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MobileViT block.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        batch_size, channels, height, width = x.shape

        # Step 1: Local feature extraction using convolution
        local_features = self.local_conv(x)

        # Step 2: Create patches for transformer processing
        # Unfold operation creates patches of size (patch_size x patch_size)
        patches = F.unfold(
            local_features,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Reshape patches for transformer input
        # patches shape: (batch_size, patch_dim, num_patches)
        # patch_dim = channels * (height/patch_size) * (width/patch_size)
        patch_dim = patches.shape[1]
        num_patches = patches.shape[2]

        # Transpose to get (batch_size, num_patches, patch_dim)
        patches = patches.transpose(1, 2)
        """
        patches is a 3-dimensional tensor
        patches consists of batch_size numbers of 2-dimensional tensors (images)
        each 2-dimensional tensor consists of num_patches numbers of 1-dimensional tensors (patches)
        each 1-dimensional tensor consists of patch_dim number of elements
        patch_dim = patch_size * patch_size * channels (height * width * channels(number of channels decided by programmer))
        """

        # Step 3: Linear projection to embedding space
        patch_embeddings = self.patch_embed(patches)

        # Step 4: Apply transformer layers for global modeling
        transformer_output = patch_embeddings
        for transformer_layer in self.transformer_layers:
            transformer_output = transformer_layer(transformer_output)

        # Step 5: Reconstruct patches from transformer output
        reconstructed_patches = self.patch_reconstruct(transformer_output)

        # Step 6: Fold patches back to spatial format
        # Transpose back to (batch_size, patch_dim, num_patches)
        reconstructed_patches = reconstructed_patches.transpose(1, 2)

        # Calculate output spatial dimensions
        output_height = height // self.patch_size
        output_width = width // self.patch_size

        # Fold patches back to spatial tensor
        spatial_output = F.fold(
            reconstructed_patches,
            output_size=(output_height, output_width),
            kernel_size=1,
            stride=1
        )

        # Step 7: Apply final convolution to adjust channels
        output = self.output_conv(spatial_output)

        return output

