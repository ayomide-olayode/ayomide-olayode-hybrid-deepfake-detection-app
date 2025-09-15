# HybridDeepfakeDetector

A PyTorch-based deepfake detection system that combines visual (face frames) and linguistic (transcribed speech) features to detect lip-sync deepfakes. The system includes a comprehensive training pipeline and an interactive Streamlit web interface for demonstrations.

## 🌟 Features

- **Hybrid Architecture**: Combines visual features (EfficientNet-B0) and text features (DistilBERT)
- **Comprehensive Pipeline**: Complete preprocessing, training, and evaluation workflow
- **Interactive Web Interface**: User-friendly Streamlit application for real-time detection
- **Robust Processing**: Handles various video formats with face detection and audio transcription
- **Detailed Analytics**: Comprehensive evaluation metrics and visualization tools

## 🏗️ Project Structure

\`\`\`
HybridDeepfakeDetector/
│
├── data/                          # Dataset directory
│   ├── train/
│   │   ├── real/                  # Real training videos
│   │   └── fake/                  # Fake training videos
│   └── test/
│       ├── real/                  # Real test videos
│       └── fake/                  # Fake test videos
│
├── src/                           # Source code
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_utils.py          # Video/audio processing utilities
│   │   └── model_utils.py         # Model utilities and metrics
│   ├── models/
│   │   ├── __init__.py
│   │   └── hybrid_model.py        # Hybrid model architecture
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── deepfake_dataset.py    # Custom PyTorch dataset
│   ├── config.py                  # Configuration settings
│   ├── preprocess.py              # Data preprocessing script
│   ├── train.py                   # Training script
│   └── validate.py                # Validation script
│
├── outputs/                       # Output directory
│   ├── models/                    # Trained model checkpoints
│   └── predictions/               # Evaluation results
│
├── app.py                         # Streamlit web application
├── run.py                         # Application launcher
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
\`\`\`

## 🚀 Quick Start

### 1. Installation

\`\`\`bash
# Clone the repository
git clone <repository-url>
cd HybridDeepfakeDetector

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
python -c "from src.config import config; config.create_directories()"
\`\`\`

### 2. Prepare Your Dataset

Place your video files in the following structure:
\`\`\`
data/
├── train/
│   ├── real/     # Real training videos (.mp4, .avi, .mov, etc.)
│   └── fake/     # Fake training videos
└── test/
    ├── real/     # Real test videos
    └── fake/     # Fake test videos
\`\`\`

### 3. Preprocess Data

\`\`\`bash
# Process all videos (extract frames and transcribe audio)
python src/preprocess.py

# Custom preprocessing options
python src/preprocess.py --data_dir ./data --output_dir ./data/processed --skip_existing
\`\`\`

### 4. Train the Model

\`\`\`bash
# Train with default settings
python src/train.py

# Custom training options
python src/train.py --epochs 100 --batch_size 32 --learning_rate 0.001
\`\`\`

### 5. Evaluate the Model

\`\`\`bash
# Evaluate on test set
python src/validate.py

# Run ablation study
python src/validate.py --ablation --save_predictions
\`\`\`

### 6. Launch Web Interface

\`\`\`bash
# Simple launch
python run.py

# Custom configuration
python run.py --port 8080 --host 0.0.0.0
\`\`\`

## 📋 Detailed Usage

### Configuration

All settings are centralized in `src/config.py`. Key parameters you might want to modify:

\`\`\`python
# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Model parameters
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 512
DROPOUT_RATE = 0.3

# Processing parameters
FRAMES_PER_SECOND = 1
WHISPER_MODEL = "base"  # tiny, base, small, medium, large
\`\`\`

### Preprocessing Options

The preprocessing script supports various options:

\`\`\`bash
python src/preprocess.py --help

Options:
  --data_dir TEXT        Directory containing train/test folders
  --output_dir TEXT      Output directory for processed data
  --skip_existing        Skip already processed videos
  --no_skip             Process all videos even if already processed
\`\`\`

**What preprocessing does:**
- Extracts frames from videos at 1 FPS
- Detects and crops faces using OpenCV
- Resizes faces to 224x224 pixels
- Transcribes audio using OpenAI Whisper
- Saves processed data for training

### Training Options

\`\`\`bash
python src/train.py --help

Options:
  --data_dir TEXT           Directory containing processed data
  --epochs INT             Number of training epochs
  --batch_size INT         Batch size for training
  --learning_rate FLOAT    Learning rate for optimizer
  --patience INT           Early stopping patience
  --resume TEXT            Path to checkpoint to resume from
  --freeze_visual          Freeze visual backbone during training
  --freeze_text            Freeze text backbone during training
\`\`\`

**Training features:**
- Automatic checkpointing and best model saving
- Early stopping to prevent overfitting
- Learning rate scheduling
- Comprehensive metrics tracking
- Training history visualization

### Validation Options

\`\`\`bash
python src/validate.py --help

Options:
  --data_dir TEXT          Directory containing processed data
  --model_path TEXT        Path to trained model checkpoint
  --batch_size INT         Batch size for evaluation
  --output_dir TEXT        Directory to save results
  --subset TEXT            Dataset subset to evaluate ('train' or 'test')
  --ablation               Perform ablation study
  --save_predictions       Save individual predictions
\`\`\`

**Validation outputs:**
- Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix and classification report
- Individual prediction results
- Visualization plots
- Ablation study results (visual-only, text-only, hybrid)

### Web Interface

The Streamlit application provides an intuitive interface for deepfake detection:

**Features:**
- Drag-and-drop video upload
- Real-time processing progress
- Visual prediction display with confidence scores
- Audio transcript viewing
- Detailed analysis results

**Launch options:**
\`\`\`bash
python run.py --help

Options:
  --port INT               Port number (default: 8501)
  --host TEXT              Host address (default: localhost)
  --debug                  Run in debug mode
  --skip-checks            Skip requirement checks
\`\`\`

## 🔧 Model Architecture

### Visual Branch
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input**: Video frames (224x224x3)
- **Processing**: Frame-level feature extraction + temporal pooling
- **Output**: 1280-dimensional visual features

### Text Branch
- **Backbone**: DistilBERT (pretrained)
- **Input**: Transcribed audio text
- **Processing**: Token-level encoding + [CLS] token extraction
- **Output**: 768-dimensional text features

### Fusion Module
- **Method**: Feature projection + concatenation
- **Architecture**: Multi-layer perceptron with dropout
- **Output**: Combined multimodal representation

### Classifier
- **Architecture**: Fully connected layers with ReLU and dropout
- **Output**: Binary classification (Real/Fake) with sigmoid activation

## 📊 Performance Metrics

The system evaluates performance using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
\`\`\`bash
# Reduce batch size
python src/train.py --batch_size 8

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
\`\`\`

**2. Missing Model File**
\`\`\`
Error: Model checkpoint not found
\`\`\`
Make sure to train the model first:
\`\`\`bash
python src/train.py
\`\`\`

**3. No Faces Detected**
\`\`\`
Error: No faces detected in the video
\`\`\`
- Ensure videos contain clear, frontal faces
- Check video quality and lighting
- Try different videos

**4. Whisper Model Download Issues**
\`\`\`bash
# Pre-download Whisper model
python -c "import whisper; whisper.load_model('base')"
\`\`\`

**5. Memory Issues During Preprocessing**
\`\`\`bash
# Process smaller batches
python src/preprocess.py --data_dir ./small_dataset
\`\`\`

### Performance Optimization

**For Training:**
- Use GPU if available (automatic detection)
- Increase batch size if you have more memory
- Use mixed precision training for faster training
- Consider freezing backbone networks initially

**For Inference:**
- Use smaller Whisper models (tiny, base) for faster transcription
- Reduce max_frames for faster processing
- Use CPU for small-scale inference

## 📁 File Formats

**Supported Video Formats:**
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)

**Output Formats:**
- Model checkpoints: PyTorch (.pth)
- Predictions: JSON (.json)
- Visualizations: PNG (.png)
- Logs: Text (.log)

## 🔬 Advanced Usage

### Custom Model Configuration

Modify `src/models/hybrid_model.py` to customize the architecture:

\`\`\`python
# Example: Different backbone models
model_config = {
    'visual_pretrained': True,
    'text_pretrained': True,
    'hidden_dim': 1024,        # Larger fusion dimension
    'dropout_rate': 0.5,       # Higher dropout
    'freeze_visual_backbone': True,  # Freeze visual features
}

model = create_model(model_config)
\`\`\`

### Custom Dataset

Extend `DeepfakeDataset` for custom data formats:

\`\`\`python
class CustomDeepfakeDataset(DeepfakeDataset):
    def _load_samples(self):
        # Custom sample loading logic
        pass
    
    def __getitem__(self, idx):
        # Custom data loading logic
        pass
\`\`\`

### Ablation Studies

Run comprehensive ablation studies:

\`\`\`bash
# Compare different modalities
python src/validate.py --ablation

# Results saved to outputs/predictions/ablation_study.json
\`\`\`

## 📈 Monitoring and Logging

The system provides comprehensive logging:

- **Training logs**: `training.log`
- **Preprocessing logs**: `preprocessing.log`
- **Validation logs**: `validation.log`
- **Training history**: `outputs/models/training_history.json`

Monitor training progress:
\`\`\`bash
# Watch training logs
tail -f training.log

# View training history
python -c "
import json
with open('outputs/models/training_history.json') as f:
    history = json.load(f)
    print(f'Best F1: {max(history[\"val_f1\"]):.4f}')
"
\`\`\`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **EfficientNet**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
- **DistilBERT**: Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
- **Whisper**: Radford, A., et al. (2022). Robust speech recognition via large-scale weak supervision.
- **OpenCV**: Bradski, G. (2000). The OpenCV Library.
- **Streamlit**: Streamlit Inc. (2019). Streamlit: The fastest way to build data apps.

## 📞 Support

For questions, issues, or contributions:

1. Check the [Issues](../../issues) page
2. Review this README and troubleshooting section
3. Create a new issue with detailed information

---

**Happy Deepfake Detection!** 🔍✨
