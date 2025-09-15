"""
Configuration file for HybridDeepfakeDetector project.
Contains all paths, hyperparameters, and settings in one place.
"""
import cv2

from pathlib import Path

class Config:
    """Centralized configuration class for the deepfake detection project."""
    
    # ============================================================================
    # PROJECT PATHS - TODO: Modify these paths according to your setup
    # ============================================================================
    
    # Root project directory
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    
    # Training data subdirectories
    TRAIN_REAL_DIR = TRAIN_DIR / "real"
    TRAIN_FAKE_DIR = TRAIN_DIR / "fake"
    
    # Test data subdirectories
    TEST_REAL_DIR = TEST_DIR / "real"
    TEST_FAKE_DIR = TEST_DIR / "fake"
    
    # Output directories
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    
    # ============================================================================
    # MODEL HYPERPARAMETERS - TODO: Adjust these for your training needs
    # ============================================================================
    
    # Training parameters
    BATCH_SIZE = 2
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 2
    PATIENCE = 10  # Early stopping patience
    
    # Model parameters
    IMAGE_SIZE = 224  # Input image size (224x224)
    MAX_TEXT_LENGTH = 512  # Maximum text sequence length
    DROPOUT_RATE = 0.3
    
    # ============================================================================
    # PREPROCESSING PARAMETERS - TODO: Modify extraction settings
    # ============================================================================
    
    # Video processing
    FRAMES_PER_SECOND = 1  # Extract 1 frame per second
    FACE_CASCADE_PATH =  cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # Audio processing
    WHISPER_MODEL = "tiny"  # Whisper model size: tiny, base, small, medium, large
    
    # ============================================================================
    # MODEL ARCHITECTURE SETTINGS
    # ============================================================================
    
    # Visual model
    VISUAL_MODEL_NAME = "efficientnet-b0"
    VISUAL_FEATURE_DIM = 1280  # EfficientNet-B0 output dimension
    
    # Text model
    TEXT_MODEL_NAME = "distilbert-base-uncased"
    TEXT_FEATURE_DIM = 768  # DistilBERT output dimension
    
    # Fusion and classification
    HIDDEN_DIM = 512
    NUM_CLASSES = 1  # Binary classification (real/fake)
    
    # ============================================================================
    # DEVICE CONFIGURATION
    # ============================================================================
    
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============================================================================
    # FILE EXTENSIONS AND FORMATS
    # ============================================================================
    
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.TRAIN_REAL_DIR,
            cls.TRAIN_FAKE_DIR,
            cls.TEST_REAL_DIR,
            cls.TEST_FAKE_DIR,
            cls.OUTPUT_DIR,
            cls.MODEL_OUTPUT_DIR,
            cls.PREDICTIONS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration settings."""
        print("=" * 60)
        print("HYBRID DEEPFAKE DETECTOR CONFIGURATION")
        print("=" * 60)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Max Text Length: {cls.MAX_TEXT_LENGTH}")
        print(f"Whisper Model: {cls.WHISPER_MODEL}")
        print("=" * 60)

# Create an instance for easy importing
config = Config()

# Auto-create directories when config is imported
if __name__ == "__main__":
    Config.create_directories()
    Config.print_config()
