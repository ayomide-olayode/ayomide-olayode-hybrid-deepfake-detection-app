import streamlit as st
import requests
import tempfile
import os
from pathlib import Path
import json
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import csv
from datetime import datetime
from io import BytesIO

# Backend API URL
BACKEND_URL = "http://localhost:8000"

# Configure Streamlit page
st.set_page_config(
    page_title="Deepfake Detection App",
    page_icon="🔍",
    layout="wide"
)

def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

def analyze_video_api(video_path, enable_video, enable_audio, enable_text, threshold, fusion_method):
    """Call backend API for analysis"""
    try:
        # Prepare request data
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {
                'enable_video': str(enable_video),
                'enable_audio': str(enable_audio),
                'enable_text': str(enable_text),
                'confidence_threshold': str(threshold),
                'fusion_method': fusion_method
            }
            
            # Call backend API
            response = requests.post(
                f"{BACKEND_URL}/analyze/multimodal",
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                st.error(f"API error: {response.status_code} - {response.text}")
                return None
            
            return response.json()
    
    except Exception as e:
        st.error(f"API communication failed: {str(e)}")
        return None

def save_results_to_csv(video_path, results, fused_results, threshold):
    """Save analysis results to CSV for chapter 4"""
    filename = "chapter4_results.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'video_file', 'duration', 
            'video_confidence', 'audio_confidence', 'text_confidence',
            'final_confidence', 'is_deepfake', 'threshold',
            'modalities_used', 'fusion_method'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'video_file': os.path.basename(video_path),
            'duration': get_video_duration(video_path),
            'video_confidence': results.get('video', {}).get('confidence', 0.5),
            'audio_confidence': results.get('audio', {}).get('confidence', 0.5),
            'text_confidence': results.get('text', {}).get('confidence', 0.5),
            'final_confidence': fused_results['confidence'],
            'is_deepfake': fused_results['confidence'] > threshold,
            'threshold': threshold,
            'modalities_used': ','.join(results.keys()),
            'fusion_method': fused_results.get('method', '')
        })

def display_video_results(results):
    """Display video-specific results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        frames_analyzed = results.get('frames_analyzed', 0)
        st.metric("Frames Analyzed", frames_analyzed)
    
    with col2:
        avg_prob = results.get('average_fake_probability', 0)
        st.metric("Avg Deepfake Prob", f"{avg_prob:.2%}")
    
    with col3:
        temporal_consistency = results.get('temporal_analysis', {}).get('temporal_consistency', 0)
        st.metric("Temporal Consistency", f"{temporal_consistency:.2%}")
    
    # Show eye blink analysis
    blink_rate = results.get('eye_blink_analysis', {}).get('blink_rate', 0)
    st.write(f"**Eye Blink Rate:** {blink_rate:.2f} (normal: 0.1-0.2)")

def display_audio_results(results):
    """Display audio-specific results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        duration = results.get('duration', 0)
        st.metric("Duration", f"{duration:.1f}s")
    
    with col2:
        naturalness = results.get('speech_analysis', {}).get('naturalness_score', 0)
        st.metric("Speech Naturalness", f"{naturalness:.2%}")
    
    with col3:
        transcription = results.get('transcription', {})
        word_count = len(transcription.get('text', '').split())
        st.metric("Words Transcribed", word_count)
    
    # Show artifact detection
    artifacts = results.get('artifacts', {})
    st.write("**Audio Artifacts Detected:**")
    st.write(f"- Phase Anomalies: {artifacts.get('phase_anomalies', 0):.2%}")
    st.write(f"- Spectral Gaps: {artifacts.get('spectral_gaps', 0):.2%}")

def display_text_results(results):
    """Display text-specific results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        word_count = results.get('word_count', 0)
        st.metric("Words Analyzed", word_count)
    
    with col2:
        semantic_score = results.get('semantic_analysis', {}).get('consistency_score', 0)
        st.metric("Semantic Consistency", f"{semantic_score:.2%}")
    
    with col3:
        clip_score = results.get('clip_analysis', {}).get('consistency_score', 0)
        st.metric("Visual-Text Alignment", f"{clip_score:.2%}")
    
    # Show detected text if available
    ocr_results = results.get('ocr_results', {})
    all_texts = ocr_results.get('all_texts', [])
    if all_texts:
        st.write("**Detected Text in Video:**")
        for i, text in enumerate(all_texts[:3]):  # Show first 3
            st.write(f"{i+1}. {text}")

def display_multimodal_results(results, fused_results, analysis_times, fusion_time, threshold, fusion_method):
    """Display comprehensive multi-modal results"""
    
    st.subheader("🎯 Multi-Modal Analysis Results")
    
    # Overall result
    final_confidence = fused_results['confidence']
    is_deepfake = final_confidence > threshold
    consistency = fused_results.get('consistency_score', 0.5)
    
    # Main result display
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if is_deepfake:
            if final_confidence > 0.8:
                st.error(f"🚨 **HIGH PROBABILITY DEEPFAKE**")
            elif final_confidence > 0.6:
                st.warning(f"⚠️ **LIKELY DEEPFAKE DETECTED**")
            else:
                st.warning(f"⚠️ **POSSIBLE DEEPFAKE**")
        else:
            if final_confidence < 0.2:
                st.success(f"✅ **HIGHLY LIKELY AUTHENTIC**")
            elif final_confidence < 0.4:
                st.success(f"✅ **LIKELY AUTHENTIC**")
            else:
                st.info(f"ℹ️ **UNCERTAIN - NEEDS REVIEW**")
    
    with col2:
        st.metric("Final Confidence", f"{final_confidence:.1%}")
        if final_confidence > 0.7:
            st.caption("🔴 High Risk")
        elif final_confidence > 0.4:
            st.caption("🟡 Medium Risk")
        else:
            st.caption("🟢 Low Risk")
    
    with col3:
        st.metric("Model Consistency", f"{consistency:.1%}")
        if consistency > 0.8:
            st.caption("🟢 High Agreement")
        elif consistency > 0.6:
            st.caption("🟡 Moderate Agreement")
        else:
            st.caption("🔴 Low Agreement")
    
    with col4:
        total_time = sum(analysis_times.values()) + fusion_time
        st.metric("Total Time", f"{total_time:.1f}s")
        st.caption(f"Fusion: {fusion_method}")
    
    # Individual model results
    st.subheader("📊 Individual Model Results")
    
    individual_confidences = {}
    if 'video' in results:
        individual_confidences['video'] = results['video'].get('confidence', 0.5)
    if 'audio' in results:
        individual_confidences['audio'] = results['audio'].get('confidence', 0.5)
    if 'text' in results:
        individual_confidences['text'] = results['text'].get('confidence', 0.5)
    
    if individual_confidences:
        # Create columns for each enabled model
        model_cols = st.columns(len(individual_confidences))
        
        for i, (modality, confidence) in enumerate(individual_confidences.items()):
            with model_cols[i]:
                # Model icon and name
                icons = {'video': '🎬', 'audio': '🎵', 'text': '📝'}
                names = {'video': 'Video Analysis', 'audio': 'Audio Analysis', 'text': 'Text Analysis'}
                
                st.metric(
                    f"{icons.get(modality, '🔍')} {names.get(modality, modality.title())}",
                    f"{confidence:.1%}",
                    f"{analysis_times.get(modality, 0):.1f}s"
                )
                
                # Individual assessment
                if confidence > 0.6:
                    st.caption("🔴 Suspicious")
                elif confidence > 0.4:
                    st.caption("🟡 Uncertain")
                else:
                    st.caption("🟢 Likely Real")
    
    # Detailed results for each modality
    for modality, result in results.items():
        if modality == 'video':
            with st.expander(f"🎬 Video Analysis Details"):
                display_video_results(result)
        elif modality == 'audio':
            with st.expander(f"🎵 Audio Analysis Details"):
                display_audio_results(result)
        elif modality == 'text':
            with st.expander(f"📝 Text Analysis Details"):
                display_text_results(result)
    
    # Fusion analysis
    with st.expander("🔗 Fusion Analysis"):
        st.write(f"**Fusion Method:** {fusion_method}")
        st.write(f"**Number of Modalities:** {len(individual_confidences)}")
        st.write(f"**Consistency Score:** {consistency:.2%}")
        
        if individual_confidences:
            st.write("**Individual Confidence Scores:**")
            for modality, confidence in individual_confidences.items():
                st.write(f"- {modality.title()}: {confidence:.2%}")
            
            # Show agreement/disagreement
            conf_values = list(individual_confidences.values())
            max_diff = max(conf_values) - min(conf_values)
            st.write(f"**Confidence Range:** {max_diff:.2%}")
            
            if max_diff < 0.2:
                st.success("✅ Models show strong agreement")
            elif max_diff < 0.4:
                st.warning("⚠️ Models show moderate disagreement")
            else:
                st.error("🚨 Models show significant disagreement - manual review recommended")
    
    # Add interpretation guide
    with st.expander("📖 How to Interpret Multi-Modal Results"):
        st.write("""
        **Multi-Modal Analysis Benefits:**
        - **Higher Accuracy**: Combines evidence from video, audio, and text
        - **Robustness**: Less likely to be fooled by single-modality attacks
        - **Confidence Assessment**: Model agreement indicates result reliability
        
        **Fusion Methods:**
        - **Weighted Average**: Balances all models with video given highest weight
        - **Maximum Confidence**: Takes the most suspicious result
        - **Consensus Voting**: Requires majority agreement for deepfake detection
        - **Advanced Fusion**: Uses cross-modal consistency and validation
        
        **Consistency Score:**
        - **High (>80%)**: All models agree - high confidence in result
        - **Medium (60-80%)**: Some disagreement - moderate confidence
        - **Low (<60%)**: Significant disagreement - manual review recommended
        """)
    
    # Download comprehensive report
    report = {
        "final_confidence": final_confidence,
        "is_deepfake": is_deepfake,
        "modality_results": results,
        "fusion_method": fusion_method,
        "analysis_times": analysis_times,
        "fusion_time": fusion_time,
        "threshold": threshold
    }
    st.download_button(
        "📄 Download Comprehensive Report",
        data=json.dumps(report, indent=2),
        file_name=f"multimodal_deepfake_analysis_report.json",
        mime="application/json"
    )
    
    # Chapter 4 report button
    if st.button("📊 Generate Chapter 4 Report"):
        generate_chapter4_report()

def generate_chapter4_report():
    """Generate performance report for Chapter 4"""
    try:
        # Load collected data
        df = pd.read_csv('chapter4_results.csv')
        
        if df.empty:
            st.warning("No analysis data available. Please analyze some videos first.")
            return
        
        # Basic statistics
        stats = {
            "total_videos": len(df),
            "deepfake_ratio": df['is_deepfake'].mean(),
            "avg_confidence": df['final_confidence'].mean(),
            "modality_usage": df['modalities_used'].value_counts().to_dict()
        }
        
        # Confusion Matrix
        y_true = df['is_deepfake'].astype(int)
        y_pred = (df['final_confidence'] > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, df['final_confidence'])
        roc_auc = auc(fpr, tpr)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # ROC Curve Plot
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic')
        ax2.legend(loc="lower right")
        
        # Confidence Distribution
        sns.histplot(df['final_confidence'], bins=20, kde=True, ax=ax3)
        ax3.set_title('Confidence Score Distribution')
        ax3.set_xlabel('Confidence Score')
        
        # Modality Performance
        modality_data = []
        for mod in ['video', 'audio', 'text']:
            if mod + '_confidence' in df.columns:
                modality_data.append({
                    'Modality': mod,
                    'Accuracy': ((df[mod + '_confidence'] > 0.5) == df['is_deepfake']).mean()
                })
        modality_df = pd.DataFrame(modality_data)
        if not modality_df.empty:
            sns.barplot(x='Modality', y='Accuracy', data=modality_df, ax=ax4)
            ax4.set_title('Modality Accuracy Comparison')
        
        plt.tight_layout()
        
        # Display in Streamlit
        st.subheader("Chapter 4: Performance Report")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Summary Statistics:**")
            st.write(f"- Total Videos Analyzed: {stats['total_videos']}")
            st.write(f"- Deepfake Detection Ratio: {stats['deepfake_ratio']:.2%}")
            st.write(f"- Average Confidence: {stats['avg_confidence']:.2f}")
            
            st.write("\n**Modality Usage:**")
            for mod, count in stats['modality_usage'].items():
                st.write(f"- {mod}: {count} videos")
        
        with col2:
            st.pyplot(fig)
        
        # Download report data
        st.download_button(
            "📊 Download Full Report Data",
            data=df.to_csv(index=False),
            file_name="deepfake_analysis_full_dataset.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Error generating Chapter 4 report: {str(e)}")

def main():
    st.title("🔍 Multi-Modal Deepfake Detection System")
    st.markdown("Upload a video file (max 30 seconds) to analyze for potential deepfake content using combined AI models.")
    
    # Configuration
    st.sidebar.header("Analysis Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    max_duration = st.sidebar.number_input("Max Video Duration (seconds)", 1, 30, 30)
    
    # Multi-modal analysis options
    st.sidebar.header("Multi-Modal Analysis")
    enable_video = st.sidebar.checkbox("Video Analysis (EfficientNet)", value=True)
    enable_audio = st.sidebar.checkbox("Audio Analysis (Whisper)", value=True) 
    enable_text = st.sidebar.checkbox("Text Analysis (BERT+CLIP)", value=True)
    
    if not any([enable_video, enable_audio, enable_text]):
        st.sidebar.error("Please enable at least one analysis method")
    
    # Fusion method selection
    st.sidebar.header("Fusion Method")
    fusion_method = st.sidebar.selectbox(
        "How to combine results",
        ["Weighted Average", "Maximum Confidence", "Consensus Voting", "Advanced Fusion"],
        index=3,
        help="Method for combining results from different modalities"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file (max 30 seconds, 100MB)"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024*1024)
        if file_size_mb > 100:
            st.error("File size too large. Please upload a video smaller than 100MB.")
            return
        
        # Save file temporarily to check duration
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Check video duration
            duration = get_video_duration(tmp_file_path)
            
            if duration > max_duration:
                st.error(f"Video duration ({duration:.1f}s) exceeds maximum allowed ({max_duration}s). Please upload a shorter video.")
                os.unlink(tmp_file_path)
                return
            
            # Display video info
            st.subheader("📹 Uploaded Video")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.video(uploaded_file)
            
            with col2:
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {file_size_mb:.2f} MB")
                st.write(f"**Duration:** {duration:.1f} seconds")
                
                enabled_models = []
                if enable_video: enabled_models.append("Video")
                if enable_audio: enabled_models.append("Audio") 
                if enable_text: enabled_models.append("Text")
                st.write(f"**Enabled Models:** {', '.join(enabled_models)}")
                st.write(f"**Fusion Method:** {fusion_method}")
            
            # Analysis button
            if st.button("🔍 Analyze Video", type="primary"):
                if not any([enable_video, enable_audio, enable_text]):
                    st.error("Please enable at least one analysis method")
                else:
                    api_result = analyze_video_api(
                        tmp_file_path, 
                        enable_video, 
                        enable_audio, 
                        enable_text, 
                        confidence_threshold, 
                        fusion_method
                    )
                    
                    if api_result:
                        # Extract results from API response
                        results = api_result.get('results', {}).get('modality_results', {})
                        fused_results = {
                            'confidence': api_result.get('confidence', 0.5),
                            'consistency_score': api_result.get('results', {}).get('consistency_score', 0.5),
                            'method': fusion_method
                        }
                        
                        # Estimate analysis times
                        analysis_times = {
                            'video': results.get('video', {}).get('processing_time', 0),
                            'audio': results.get('audio', {}).get('processing_time', 0),
                            'text': results.get('text', {}).get('processing_time', 0)
                        }
                        fusion_time = api_result.get('analysis_time_seconds', 0) - sum(analysis_times.values())
                        
                        # Display results
                        display_multimodal_results(
                            results, 
                            fused_results, 
                            analysis_times, 
                            fusion_time, 
                            confidence_threshold, 
                            fusion_method
                        )
                        
                        # Save results for Chapter 4
                        save_results_to_csv(tmp_file_path, results, fused_results, confidence_threshold)
        
        finally:
            # Clean up
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()