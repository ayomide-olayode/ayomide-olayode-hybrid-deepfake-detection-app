#!/bin/bash

# Sample data download script for HybridDeepfakeDetector
# This script demonstrates how to organize your dataset

echo "=================================================="
echo "🔍 HYBRID DEEPFAKE DETECTOR - SAMPLE DATA SETUP"
echo "=================================================="

# Create data directories
echo "Creating data directories..."
mkdir -p data/train/real
mkdir -p data/train/fake
mkdir -p data/test/real
mkdir -p data/test/fake

echo "✅ Data directories created"

# Create sample README files
cat > data/README.md << 'EOF'
# Dataset Directory

Place your video files in the following structure:
