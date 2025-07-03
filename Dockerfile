FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads logs

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    echo "Starting FastAPI server..."\n\
    uvicorn main:app --host 0.0.0.0 --port 8000\n\
elif [ "$1" = "streamlit" ]; then\n\
    echo "Starting Streamlit app..."\n\
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0\n\
else\n\
    echo "Usage: docker run <image> [api|streamlit]"\n\
    echo "  api       - Start FastAPI backend server"\n\
    echo "  streamlit - Start Streamlit web interface"\n\
    exit 1\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["./start.sh", "streamlit"]
