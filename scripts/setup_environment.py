"""
Environment setup script for HybridDeepfakeDetector.
Helps users set up the project environment and download required models.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages."""
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        return False

def download_whisper_model():
    """Pre-download Whisper model to avoid runtime delays."""
    logger.info("Downloading Whisper model...")
    try:
        import whisper
        whisper.load_model("base")
        logger.info("✅ Whisper model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download Whisper model: {e}")
        return False

def create_directories():
    """Create necessary project directories."""
    logger.info("Creating project directories...")
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    
    try:
        from config import config
        config.create_directories()
        logger.info("✅ Project directories created")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create directories: {e}")
        return False

def check_opencv():
    """Check OpenCV installation and cascade files."""
    logger.info("Checking OpenCV installation...")
    try:
        import cv2
        
        # Check cascade file
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            logger.info("✅ OpenCV and cascade files are available")
            return True
        else:
            logger.error("❌ OpenCV cascade files not found")
            return False
    except ImportError:
        logger.error("❌ OpenCV not installed")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("🔍 HYBRID DEEPFAKE DETECTOR - ENVIRONMENT SETUP")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Check OpenCV
    if not check_opencv():
        success = False
    
    # Download Whisper model
    if not download_whisper_model():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Place your dataset in the data/ directory")
        print("2. Run preprocessing: python src/preprocess.py")
        print("3. Train the model: python src/train.py")
        print("4. Launch the app: python run.py")
    else:
        print("❌ SETUP FAILED!")
        print("Please check the error messages above and resolve the issues.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
