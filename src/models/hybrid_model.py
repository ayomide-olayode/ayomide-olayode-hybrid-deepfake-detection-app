"""
Hybrid deepfake detection model combining visual and textual features.
Uses EfficientNet-B0 for visual features and DistilBERT for text features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig
from efficientnet_pytorch import EfficientNet
from typing import Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualBranch(nn.Module):
    """Visual feature extraction branch using EfficientNet-B0."""
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize visual branch.
        
        Args:
            pretrained: Whether to use pretrained EfficientNet weights
            freeze_backbone: Whether to freeze backbone parameters
        """
        super(VisualBranch, self).__init__()
        
        # Load EfficientNet-B0
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
            logger.info("Loaded pretrained EfficientNet-B0")
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b0')
            logger.info("Loaded EfficientNet-B0 without pretrained weights")
        
        # Get feature dimension (EfficientNet-B0 outputs 1280 features)
        self.feature_dim = self.backbone._fc.in_features
        
        # Remove the final classification layer
        self.backbone._fc = nn.Identity()
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Frozen EfficientNet backbone parameters")
        
        # Add adaptive pooling to handle variable number of frames
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through visual branch.
        
        Args:
            x: Input tensor of shape (batch_size, num_frames, channels, height, width)
            
        Returns:
            Visual features of shape (batch_size, feature_dim)
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features from all frames
        frame_features = self.backbone(x)  # (batch_size * num_frames, feature_dim)
        
        # Reshape back to separate frames
        frame_features = frame_features.view(batch_size, num_frames, self.feature_dim)
        
        # Average pool across frames to get single representation per video
        # Transpose for adaptive pooling: (batch_size, feature_dim, num_frames)
        frame_features = frame_features.transpose(1, 2)
        pooled_features = self.adaptive_pool(frame_features).squeeze(-1)  # (batch_size, feature_dim)
        
        return pooled_features

class TextBranch(nn.Module):
    """Text feature extraction branch using DistilBERT."""
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Initialize text branch.
        
        Args:
            pretrained: Whether to use pretrained DistilBERT weights
            freeze_backbone: Whether to freeze backbone parameters
        """
        super(TextBranch, self).__init__()
        
        # Load DistilBERT
        if pretrained:
            self.backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')
            logger.info("Loaded pretrained DistilBERT")
        else:
            config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
            self.backbone = DistilBertModel(config)
            logger.info("Loaded DistilBERT without pretrained weights")
        
        # Get feature dimension (DistilBERT outputs 768 features)
        self.feature_dim = self.backbone.config.hidden_size
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Frozen DistilBERT backbone parameters")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through text branch.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            
        Returns:
            Text features of shape (batch_size, feature_dim)
        """
        # Get DistilBERT outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        text_features = outputs.last_hidden_state[:, 0, :]  # (batch_size, feature_dim)
        
        return text_features

class FusionModule(nn.Module):
    """Fusion module to combine visual and text features."""
    
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int, dropout_rate: float = 0.3):
        """
        Initialize fusion module.
        
        Args:
            visual_dim: Dimension of visual features
            text_dim: Dimension of text features
            hidden_dim: Hidden dimension for fusion layers
            dropout_rate: Dropout rate for regularization
        """
        super(FusionModule, self).__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Project visual and text features to same dimension
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Fusion layers
        # Concatenated features dimension: hidden_dim + hidden_dim = 2 * hidden_dim
        self.fusion_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_dim = hidden_dim // 2
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion module.
        
        Args:
            visual_features: Visual features of shape (batch_size, visual_dim)
            text_features: Text features of shape (batch_size, text_dim)
            
        Returns:
            Fused features of shape (batch_size, output_dim)
        """
        # Project features to same dimension
        visual_proj = self.visual_projection(visual_features)  # (batch_size, hidden_dim)
        text_proj = self.text_projection(text_features)        # (batch_size, hidden_dim)
        
        # Concatenate projected features
        fused_features = torch.cat([visual_proj, text_proj], dim=1)  # (batch_size, 2 * hidden_dim)
        
        # Pass through fusion layers
        output = self.fusion_layers(fused_features)  # (batch_size, output_dim)
        
        return output

class HybridDeepfakeDetector(nn.Module):
    """
    Hybrid deepfake detection model combining visual and textual modalities.
    """
    
    def __init__(self, 
                 visual_pretrained: bool = True,
                 text_pretrained: bool = True,
                 freeze_visual_backbone: bool = False,
                 freeze_text_backbone: bool = False,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.3,
                 num_classes: int = 1):
        """
        Initialize hybrid deepfake detector.
        
        Args:
            visual_pretrained: Whether to use pretrained visual backbone
            text_pretrained: Whether to use pretrained text backbone
            freeze_visual_backbone: Whether to freeze visual backbone
            freeze_text_backbone: Whether to freeze text backbone
            hidden_dim: Hidden dimension for fusion module
            dropout_rate: Dropout rate for regularization
            num_classes: Number of output classes (1 for binary classification)
        """
        super(HybridDeepfakeDetector, self).__init__()
        
        # Initialize branches
        self.visual_branch = VisualBranch(
            pretrained=visual_pretrained,
            freeze_backbone=freeze_visual_backbone
        )
        
        self.text_branch = TextBranch(
            pretrained=text_pretrained,
            freeze_backbone=freeze_text_backbone
        )
        
        # Initialize fusion module
        self.fusion_module = FusionModule(
            visual_dim=self.visual_branch.feature_dim,
            text_dim=self.text_branch.feature_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_module.output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
        
        self.num_classes = num_classes
        
        logger.info(f"Initialized HybridDeepfakeDetector with {self._count_parameters():,} parameters")
    
    def forward(self, visual_input: torch.Tensor, text_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            visual_input: Visual input tensor of shape (batch_size, num_frames, channels, height, width)
            text_input: Dictionary containing 'input_ids' and 'attention_mask'
            
        Returns:
            Predictions of shape (batch_size, num_classes)
        """
        # Extract visual features
        visual_features = self.visual_branch(visual_input)
        
        # Extract text features
        text_features = self.text_branch(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        )
        
        # Fuse features
        fused_features = self.fusion_module(visual_features, text_features)
        
        # Final classification
        predictions = self.classifier(fused_features)
        
        return predictions
    
    def forward_visual_only(self, visual_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using only visual features (for ablation studies).
        
        Args:
            visual_input: Visual input tensor
            
        Returns:
            Predictions based on visual features only
        """
        visual_features = self.visual_branch(visual_input)
        
        # Simple classifier for visual-only prediction
        if not hasattr(self, 'visual_only_classifier'):
            self.visual_only_classifier = nn.Sequential(
                nn.Linear(self.visual_branch.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
                nn.Sigmoid() if self.num_classes == 1 else nn.Softmax(dim=1)
            ).to(visual_input.device)
        
        return self.visual_only_classifier(visual_features)
    
    def forward_text_only(self, text_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using only text features (for ablation studies).
        
        Args:
            text_input: Dictionary containing text input
            
        Returns:
            Predictions based on text features only
        """
        text_features = self.text_branch(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        )
        
        # Simple classifier for text-only prediction
        if not hasattr(self, 'text_only_classifier'):
            self.text_only_classifier = nn.Sequential(
                nn.Linear(self.text_branch.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
                nn.Sigmoid() if self.num_classes == 1 else nn.Softmax(dim=1)
            ).to(text_input['input_ids'].device)
        
        return self.text_only_classifier(text_features)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions for each component."""
        return {
            'visual_features': self.visual_branch.feature_dim,
            'text_features': self.text_branch.feature_dim,
            'fused_features': self.fusion_module.output_dim,
            'total_parameters': self._count_parameters()
        }

def create_model(config_dict: Optional[Dict] = None) -> HybridDeepfakeDetector:
    """
    Create hybrid deepfake detector model with configuration.
    
    Args:
        config_dict: Configuration dictionary (optional)
        
    Returns:
        Initialized HybridDeepfakeDetector model
    """
    # Default configuration
    default_config = {
        'visual_pretrained': True,
        'text_pretrained': True,
        'freeze_visual_backbone': False,
        'freeze_text_backbone': False,
        'hidden_dim': 512,
        'dropout_rate': 0.3,
        'num_classes': 1
    }
    
    # Update with provided config
    if config_dict:
        default_config.update(config_dict)
    
    # Create model
    model = HybridDeepfakeDetector(**default_config)
    
    # Print model information
    feature_dims = model.get_feature_dimensions()
    logger.info("Model Feature Dimensions:")
    for key, value in feature_dims.items():
        logger.info(f"  {key}: {value}")
    
    return model

# TODO: Modify these parameters based on your requirements
def get_model_config():
    """
    Get model configuration. Modify these parameters as needed.
    
    Returns:
        Dictionary with model configuration
    """
    return {
        'visual_pretrained': True,      # Use pretrained EfficientNet
        'text_pretrained': True,        # Use pretrained DistilBERT
        'freeze_visual_backbone': False, # Set to True to freeze visual backbone
        'freeze_text_backbone': False,   # Set to True to freeze text backbone
        'hidden_dim': 512,              # Hidden dimension for fusion
        'dropout_rate': 0.3,            # Dropout rate for regularization
        'num_classes': 1                # 1 for binary classification
    }

if __name__ == "__main__":
    # Test model creation
    model = create_model(get_model_config())
    
    # Test forward pass with dummy data
    batch_size = 2
    num_frames = 5
    
    # Dummy visual input
    visual_input = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    # Dummy text input
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(visual_input, text_input)
        print(f"Output shape: {output.shape}")
        print(f"Output values: {output}")
