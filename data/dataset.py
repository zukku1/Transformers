"""
Dataset and Transform Components for Deepfake Detection

This module provides:
1. Custom dataset class for loading deepfake detection data
2. Comprehensive data augmentation strategies
3. Preprocessing pipelines optimized for MobileViT
4. Handling of various image sizes and formats

The dataset assumes a folder structure like:
dataset/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
"""

import os
import random
import tempfile
from typing import Optional, Tuple, List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class DeepfakeDataset(Dataset):
    """
    Custom dataset class for deepfake detection.

    This dataset handles loading images from organized folders and applies
    appropriate transformations for training or evaluation.

    Args:
        root_dir: Root directory containing 'real' and 'fake' subdirectories
        transform: Torchvision transforms to apply to images
        max_samples: Maximum number of samples to load (for debugging)
        cache_size: Number of images to keep in memory cache
    """

    def __init__(self,
                 root_dir: str,
                 transform: Optional[callable] = None,
                 max_samples: Optional[int] = None,
                 cache_size: int = 1000):

        self.root_dir = root_dir
        self.transform = transform
        self.cache_size = cache_size
        self.image_cache = {}  # Simple LRU cache for frequently accessed images

        # Find all image files in real and fake directories
        self.samples = []
        self._load_samples(max_samples)

        # Class mapping: real=0, fake=1
        self.class_to_idx = {'real': 0, 'fake': 1}
        self.idx_to_class = {0: 'real', 1: 'fake'}

        print(f"Loaded {len(self.samples)} samples from {root_dir}")
        print(f"Real samples: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"Fake samples: {sum(1 for _, label in self.samples if label == 1)}")

    def _load_samples(self, max_samples: Optional[int]):
        """
        Load all valid image file paths and their labels.

        Args:
            max_samples: Maximum number of samples to load
        """
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # Load real images (label = 0)
        real_dir = os.path.join(self.root_dir, 'real')
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    file_path = os.path.join(real_dir, filename)
                    if os.path.isfile(file_path):
                        self.samples.append((file_path, 0))

        # Load fake images (label = 1)
        fake_dir = os.path.join(self.root_dir, 'fake')
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    file_path = os.path.join(fake_dir, filename)
                    if os.path.isfile(file_path):
                        self.samples.append((file_path, 1))

        # Shuffle samples for better training distribution
        random.shuffle(self.samples)

        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load image with caching and error handling.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object
        """
        # Check cache first
        if image_path in self.image_cache:
            return self.image_cache[image_path].copy()

        try:
            # Load image using PIL for better format support
            image = Image.open(image_path).convert('RGB')

            # Add to cache if there's space
            if len(self.image_cache) < self.cache_size:
                self.image_cache[image_path] = image.copy()

            return image

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, label)
        """
        # Get image path and label
        image_path, label = self.samples[idx]

        # Load image
        image = self._load_image(image_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.

        Returns:
            Tensor of class weights for loss function
        """
        # Count samples per class
        class_counts = [0, 0]
        for _, label in self.samples:
            class_counts[label] += 1

        # Calculate inverse frequency weights
        total_samples = len(self.samples)
        weights = [total_samples / (2 * count) for count in class_counts]

        return torch.FloatTensor(weights)


class AdvancedAugmentations:
    """
    Advanced augmentation techniques specifically for deepfake detection.

    This class provides sophisticated augmentations that help the model
    generalize better and become more robust to various types of deepfakes.
    """

    @staticmethod
    def get_training_transforms(image_size: int = 224,
                                augment_prob: float = 0.8) -> callable:
        """
        Get comprehensive training augmentations.

        Args:
            image_size: Target image size
            augment_prob: Probability of applying augmentations

        Returns:
            Albumentations transform pipeline
        """
        return A.Compose([
            # Resize to slightly larger size for random cropping
            A.Resize(int(image_size * 1.2), int(image_size * 1.2)),

            # Geometric augmentations
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.8
            ),

            # Horizontal flip (common for face images)
            A.HorizontalFlip(p=0.5),

            # Rotation with small angles to maintain face structure
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.3),

            # Color and lighting augmentations
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.6),

            # Lighting changes
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),

            # Noise and blur (common in deepfakes)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.3),

            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # Compression artifacts (common in deepfakes)
            A.ImageCompression(quality_range=(60, 100), p=0.3),

            # Advanced augmentations for deepfake robustness
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=5, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                A.OpticalDistortion(distort_limit=(0.05, 0.15), p=1.0),
            ], p=0.2),

            # Cutout/random erasing
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0, image_size // 8),
                hole_width_range=(0, image_size // 8),
                fill=0,
                p=0.3
            ),

            # Normalize to ImageNet statistics (important for pretrained models)
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            # Convert to PyTorch tensor
            ToTensorV2()
        ])

    @staticmethod
    def get_validation_transforms(image_size: int = 224) -> callable:
        """
        Get validation/test transforms without augmentation.

        Args:
            image_size: Target image size

        Returns:
            Albumentations transform pipeline
        """
        return A.Compose([
            # Resize to target size
            A.Resize(image_size, image_size),

            # Normalize to ImageNet statistics
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            # Convert to PyTorch tensor
            ToTensorV2()
        ])

    @staticmethod
    def get_test_time_augmentation_transforms(image_size: int = 224) -> List[callable]:
        """
        Get multiple transforms for test-time augmentation.

        Test-time augmentation applies multiple transformations to the same image
        and averages the predictions for more robust results.

        Args:
            image_size: Target image size

        Returns:
            List of transform pipelines
        """
        base_transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]

        # Different TTA variations
        tta_transforms = [A.Compose(base_transforms), A.Compose([
            A.HorizontalFlip(p=1.0),
            *base_transforms
        ])]

        # Original image

        # Horizontal flip

        # Slight rotations
        for angle in [-5, 5]:
            tta_transforms.append(A.Compose([
                A.Rotate(limit=(angle, angle), border_mode=cv2.BORDER_REFLECT_101, p=1.0),
                *base_transforms
            ]))

        # Brightness variations
        for brightness in [-0.1, 0.1]:
            tta_transforms.append(A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(brightness, brightness), contrast_limit=0, p=1.0),
                *base_transforms
            ]))

        return tta_transforms


class FaceAwareTransforms:
    """
    Face-aware transformations that preserve facial structure.

    These transforms are specifically designed for face images and deepfake
    detection, ensuring that important facial features are preserved while
    still providing useful augmentation.
    """

    @staticmethod
    def get_face_preserving_transforms(image_size: int = 224) -> callable:
        """
        Get transforms that preserve facial structure.

        Args:
            image_size: Target image size

        Returns:
            Transform pipeline optimized for face images
        """
        return A.Compose([
            # Resize with aspect ratio preservation
            A.LongestMaxSize(max_size=int(image_size * 1.1)),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.CenterCrop(height=image_size, width=image_size),

            # Gentle geometric transforms
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.3),

            # Face-friendly color adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.4
            ),

            # Subtle hue/saturation changes
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),

            # Minimal noise (deepfakes often have artifacts)
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),

            # Very light blur (compression artifacts)
            A.GaussianBlur(blur_limit=(1, 2), p=0.1),

            # Normalize and convert to tensor
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


class MultiScaleDataset(Dataset):
    """
    Dataset that provides images at multiple scales for multiscale training.

    This is useful for training models that process features at different
    resolutions, which can be beneficial for deepfake detection.

    Args:
        base_dataset: Base dataset to wrap
        scales: List of scales to generate
        base_size: Base image size
    """

    def __init__(self,
                 base_dataset: DeepfakeDataset,
                 scales=None,
                 base_size: int = 224):

        if scales is None:
            scales = [0.8, 1.0, 1.2]
        self.base_dataset = base_dataset
        self.scales = scales
        self.base_size = base_size

        # Create transforms for each scale
        self.scale_transforms = {}
        for scale in scales:
            size = int(base_size * scale)
            self.scale_transforms[scale] = A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        """
        Get multiscale versions of a single image.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (list_of_scaled_images, label)
        """
        # Get original image and label (without transforms)
        image_path, label = self.base_dataset.samples[idx]
        image = self.base_dataset._load_image(image_path)

        # Convert PIL to numpy for albumentations
        image_np = np.array(image)

        # Apply transforms at different scales
        scaled_images = []
        for scale in self.scales:
            transformed = self.scale_transforms[scale](image=image_np)
            scaled_images.append(transformed['image'])

        return scaled_images, label


def create_data_loaders(train_dir: str,
                        val_dir: str,
                        test_dir: Optional[str] = None,
                        batch_size: int = 32,
                        image_size: int = 224,
                        num_workers: int = 4,
                        augmentation_type: str = 'advanced') -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        test_dir: Directory containing test data (optional)
        batch_size: Batch size for training
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        augmentation_type: Type of augmentation ('basic', 'advanced', 'face_aware')

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    # Choose augmentation strategy
    global test_dataset
    if augmentation_type == 'advanced':
        train_transform = AdvancedAugmentations.get_training_transforms(image_size)
    elif augmentation_type == 'face_aware':
        train_transform = FaceAwareTransforms.get_face_preserving_transforms(image_size)
    else:  # basic
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    # Validation transform (no augmentation)
    val_transform = AdvancedAugmentations.get_validation_transforms(image_size)

    # Create datasets
    train_dataset = DeepfakeDataset(train_dir, transform=train_transform)
    val_dataset = DeepfakeDataset(val_dir, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True  # Ensure consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create test loader if test directory is provided
    test_loader = None
    if test_dir and os.path.exists(test_dir):
        test_dataset = DeepfakeDataset(test_dir, transform=val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # Print dataset statistics
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_dataset)}")

    # Calculate and print class weights for balanced training
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")

    return train_loader, val_loader, test_loader


def visualize_augmentations(dataset: DeepfakeDataset,
                            num_samples: int = 8,
                            save_path: Optional[str] = None):
    """
    Visualize the effect of data augmentations.

    Args:
        dataset: Dataset with augmentations
        num_samples: Number of samples to visualize
        save_path: Path to save visualization (optional)
    """

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_samples):
        # Get a sample
        image, label = dataset[i]

        # Convert tensor back to image
        if isinstance(image, torch.Tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)

            # Convert to numpy
            image = image.permute(1, 2, 0).numpy()

        # Plot
        axes[i].imshow(image)
        axes[i].set_title(f"{'Real' if label == 0 else 'Fake'}")
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Test the dataset and transforms
    print("Testing dataset and transforms...")

    # Create dummy directory structure for testing

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data structure
        train_dir = os.path.join(temp_dir, 'train')
        os.makedirs(os.path.join(train_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'fake'), exist_ok=True)

        # Create dummy images (in practice, these would be real images)
        for i in range(5):
            dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            dummy_img.save(os.path.join(train_dir, 'real', f'real_{i}.jpg'))
            dummy_img.save(os.path.join(train_dir, 'fake', f'fake_{i}.jpg'))

        # Test dataset creation
        transform = AdvancedAugmentations.get_training_transforms(224)
        dataset = DeepfakeDataset(train_dir, transform=transform)

        print(f"Dataset size: {len(dataset)}")
        print(f"Class weights: {dataset.get_class_weights()}")

        # Test data loading
        sample_image, sample_label = dataset[0]
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample label: {sample_label}")

        print("Dataset and transforms test completed successfully!")