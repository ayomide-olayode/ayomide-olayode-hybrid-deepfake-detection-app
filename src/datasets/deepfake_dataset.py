"""
Custom PyTorch Dataset for deepfake detection.
Loads preprocessed frames and transcripts for training and evaluation.
"""

import os
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    """
    Custom Dataset for loading preprocessed deepfake detection data.
    Handles both visual frames and text transcripts.
    """
    
    def __init__(self, 
                 data_dir: str,
                 subset: str = 'train',
                 max_frames: int = 10,
                 max_text_length: int = 512,
                 image_size: int = 224,
                 tokenizer_name: str = 'distilbert-base-uncased',
                 augment: bool = True):
        """
        Initialize DeepfakeDataset.
        
        Args:
            data_dir: Root directory containing processed data
            subset: Dataset subset ('train' or 'test')
            max_frames: Maximum number of frames to load per video
            max_text_length: Maximum text sequence length
            image_size: Target image size (assumes square images)
            tokenizer_name: Name of the tokenizer to use
            augment: Whether to apply data augmentation (only for training)
        """
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.max_frames = max_frames
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.augment = augment and (subset == 'train')
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Loaded tokenizer: {tokenizer_name}")
        
        # Define image transforms
        self._setup_transforms()
        
        # Load dataset samples
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {subset} set")
        
        # Print class distribution
        self._print_class_distribution()
    
    def _setup_transforms(self):
        """Setup image transforms for training and validation."""
        
        # Base transforms (always applied)
        base_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ]
        
        if self.augment:
            # Training transforms with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),  # Slightly larger
                transforms.RandomCrop((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info("Using augmented transforms for training")
        else:
            # Validation transforms (no augmentation)
            self.transform = transforms.Compose(base_transforms)
            logger.info("Using standard transforms for validation")
    
    def _load_samples(self) -> List[Dict]:
        """
        Load all samples from the dataset directory.
        
        Returns:
            List of sample dictionaries containing paths and labels
        """
        samples = []
        
        # Define class mapping
        class_mapping = {'real': 0, 'fake': 1}
        
        subset_dir = self.data_dir / self.subset
        
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        
        for class_name in ['real', 'fake']:
            class_dir = subset_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Find all video directories (ending with _frames)
            frame_dirs = [d for d in class_dir.iterdir() if d.is_dir() and d.name.endswith('_frames')]
            
            for frame_dir in frame_dirs:
                # Get corresponding transcript file
                video_name = frame_dir.name.replace('_frames', '')
                transcript_path = class_dir / f"{video_name}_transcript.txt"
                transcript_json_path = class_dir / f"{video_name}_transcript.json"
                
                # Check if transcript exists
                if not transcript_path.exists():
                    logger.warning(f"Transcript not found for {video_name}, skipping")
                    continue
                
                # Get frame files
                frame_files = self._get_frame_files(frame_dir)
                
                if len(frame_files) == 0:
                    logger.warning(f"No frames found in {frame_dir}, skipping")
                    continue
                
                # Create sample entry
                sample = {
                    'video_name': video_name,
                    'class_name': class_name,
                    'label': class_mapping[class_name],
                    'frame_dir': str(frame_dir),
                    'frame_files': frame_files,
                    'transcript_path': str(transcript_path),
                    'transcript_json_path': str(transcript_json_path) if transcript_json_path.exists() else None,
                    'num_frames': len(frame_files)
                }
                
                samples.append(sample)
        
        return samples
    
    def _get_frame_files(self, frame_dir: Path) -> List[str]:
        """
        Get sorted list of frame files from directory.
        
        Args:
            frame_dir: Directory containing frame files
            
        Returns:
            Sorted list of frame file paths
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        frame_files = []
        
        for ext in image_extensions:
            frame_files.extend(frame_dir.glob(f"*{ext}"))
            frame_files.extend(frame_dir.glob(f"*{ext.upper()}"))
        
        # Sort files to ensure consistent ordering
        frame_files = sorted([str(f) for f in frame_files])
        
        return frame_files
    
    def _print_class_distribution(self):
        """Print class distribution statistics."""
        class_counts = {}
        total_frames = 0
        
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_frames += sample['num_frames']
        
        logger.info(f"Class distribution for {self.subset}:")
        for class_name, count in class_counts.items():
            percentage = (count / len(self.samples)) * 100
            logger.info(f"  {class_name}: {count} videos ({percentage:.1f}%)")
        
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Average frames per video: {total_frames / len(self.samples):.1f}")
    
    def _load_frames(self, frame_files: List[str]) -> torch.Tensor:
        """
        Load and process frames from files.
        
        Args:
            frame_files: List of frame file paths
            
        Returns:
            Tensor of shape (num_frames, channels, height, width)
        """
        # Sample frames if we have more than max_frames
        if len(frame_files) > self.max_frames:
            # Evenly sample frames
            indices = np.linspace(0, len(frame_files) - 1, self.max_frames, dtype=int)
            selected_files = [frame_files[i] for i in indices]
        else:
            selected_files = frame_files
        
        frames = []
        
        for frame_path in selected_files:
            try:
                # Load image
                image = Image.open(frame_path).convert('RGB')
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                frames.append(image)
                
            except Exception as e:
                logger.warning(f"Error loading frame {frame_path}: {e}")
                # Create a black frame as fallback
                black_frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(black_frame)
        
        # Pad with black frames if we have fewer than max_frames
        while len(frames) < self.max_frames:
            black_frame = torch.zeros(3, self.image_size, self.image_size)
            frames.append(black_frame)
        
        # Stack frames
        frames_tensor = torch.stack(frames)  # (num_frames, channels, height, width)
        
        return frames_tensor
    
    def _load_transcript(self, transcript_path: str, transcript_json_path: Optional[str] = None) -> str:
        """
        Load transcript text from file.
        
        Args:
            transcript_path: Path to transcript text file
            transcript_json_path: Path to transcript JSON file (optional)
            
        Returns:
            Transcript text string
        """
        try:
            # Try to load from JSON first (contains more structured data)
            if transcript_json_path and os.path.exists(transcript_json_path):
                with open(transcript_json_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    text = transcript_data.get('text', '')
            else:
                # Load from plain text file
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Error loading transcript {transcript_path}: {e}")
            return ""  # Return empty string as fallback
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text using DistilBERT tokenizer.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing input_ids and attention_mask tensors
        """
        # Handle empty text
        if not text.strip():
            text = "[EMPTY]"  # Use special token for empty transcripts
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, Union[str, int]]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple containing:
            - frames: Tensor of shape (max_frames, channels, height, width)
            - text_data: Dictionary with tokenized text
            - label: Label tensor
            - metadata: Dictionary with sample metadata
        """
        sample = self.samples[idx]
        
        # Load frames
        frames = self._load_frames(sample['frame_files'])
        
        # Load and tokenize transcript
        transcript_text = self._load_transcript(
            sample['transcript_path'], 
            sample.get('transcript_json_path')
        )
        text_data = self._tokenize_text(transcript_text)
        
        # Create label tensor
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        # Create metadata
        metadata = {
            'video_name': sample['video_name'],
            'class_name': sample['class_name'],
            'num_frames': sample['num_frames'],
            'transcript_length': len(transcript_text)
        }
        
        return frames, text_data, label, metadata
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.
        
        Returns:
            Tensor containing class weights
        """
        class_counts = {}
        
        for sample in self.samples:
            label = sample['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.samples)
        num_classes = len(class_counts)
        
        # Calculate weights (inverse frequency)
        weights = []
        for class_idx in sorted(class_counts.keys()):
            weight = total_samples / (num_classes * class_counts[class_idx])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

def create_data_loaders(data_dir: str, 
                       batch_size: int = 16,
                       max_frames: int = 10,
                       max_text_length: int = 512,
                       image_size: int = 224,
                       num_workers: int = 4) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test data loaders.
    
    Args:
        data_dir: Root directory containing processed data
        batch_size: Batch size for data loaders
        max_frames: Maximum frames per video
        max_text_length: Maximum text sequence length
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=data_dir,
        subset='train',
        max_frames=max_frames,
        max_text_length=max_text_length,
        image_size=image_size,
        augment=True
    )
    
    test_dataset = DeepfakeDataset(
        data_dir=data_dir,
        subset='test',
        max_frames=max_frames,
        max_text_length=max_text_length,
        image_size=image_size,
        augment=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for training
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, test_loader

# TODO: Modify these parameters based on your data and requirements
def get_dataset_config():
    """
    Get dataset configuration. Modify these parameters as needed.
    
    Returns:
        Dictionary with dataset configuration
    """
    return {
        'max_frames': 10,           # Maximum frames to load per video
        'max_text_length': 512,     # Maximum text sequence length
        'image_size': 224,          # Target image size (224x224)
        'batch_size': 16,           # Batch size for training
        'num_workers': 4            # Number of data loading workers
    }

if __name__ == "__main__":
    # Test dataset creation
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import config
    
    # Test with processed data directory
    processed_data_dir = config.DATA_DIR / "processed"
    
    if processed_data_dir.exists():
        try:
            train_loader, test_loader = create_data_loaders(
                data_dir=str(processed_data_dir),
                **get_dataset_config()
            )
            
            # Test loading a batch
            for batch_idx, (frames, text_data, labels, metadata) in enumerate(train_loader):
                print(f"Batch {batch_idx}:")
                print(f"  Frames shape: {frames.shape}")
                print(f"  Input IDs shape: {text_data['input_ids'].shape}")
                print(f"  Attention mask shape: {text_data['attention_mask'].shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Metadata: {metadata}")
                break
                
        except Exception as e:
            logger.error(f"Error testing dataset: {e}")
            logger.info("Make sure to run preprocessing first to create processed data")
    else:
        logger.info(f"Processed data directory not found: {processed_data_dir}")
        logger.info("Run preprocessing first to create the dataset")
