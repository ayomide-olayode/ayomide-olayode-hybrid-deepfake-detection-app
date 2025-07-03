# 🧠 Hybrid Deepfake Detection System

This project implements a **hybrid AI-based deepfake detection system** that fuses **visual**, **audio**, and **textual** modalities to identify deepfakes. It combines state-of-the-art pretrained models with a custom-built fusion pipeline and provides both an API and web interface for interaction.

---

## 🔍 Features

- ✅ **Visual Processing** using EfficientNet
- 🧏‍♂️ **Audio Transcription** with Whisper
- 📝 **Text Extraction** via EasyOCR
- 🔗 **Cross-modal Semantic Analysis** with BERT & CLIP
- 🎯 **Lip-sync Inconsistency Detection** using SyncNet
- 🔄 **Graph-based Fusion** using GNN and Cross-Modal Attention
- 📊 **Evaluation Metrics & Reports**
- 🌐 **Streamlit UI + FastAPI Backend**
- 🐳 **Docker Support**

---

## 📁 Project Structure

```plaintext
Deepfake detection app/
├── app.py                  # Streamlit frontend
├── main.py                 # FastAPI backend
├── evaluate.py             # Model evaluation
├── modules/                # Core processing modules
│   ├── video_processor.py
│   ├── audio_processor.py
│   ├── text_processor.py
│   ├── fusion.py
│   ├── detector.py
│   ├── metrics.py
│   └── generatereport.py
├── Dockerfile              # Docker build file
├── requirements.txt        # Dependency list

---

## 🚀 Getting Started
1️⃣ Install Dependencies
Make sure you’re using Python 3.10+:
```plaintext
pip install -r requirements.txt
2️⃣ Run the Streamlit App
```plaintext
streamlit run app.py
3️⃣ Run the FastAPI Backend
```plaintext
uvicorn main:app --reload

---

##🧪 Evaluation
To test the model and generate performance metrics:
```plaintext
python evaluate.py

---

##🤖 Models Used
```plaintext
| Modality | Model(s)                   |
| -------- | -------------------------- |
| Visual   | EfficientNetB4             |
| Audio    | Whisper, SyncNet           |
| Text     | EasyOCR, BERT              |
| Fusion   | Cross-Modal Attention, GNN |
| Semantic | CLIP                       |

---

##✍️ Author
Built by Ayomide Olayode for a final year research project on AI-driven deepfake detection.

