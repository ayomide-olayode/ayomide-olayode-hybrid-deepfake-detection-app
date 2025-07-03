import os
import json
import cv2
import numpy as np
import time
from modules.video_processor import VideoProcessor
from modules.audio_processor import AudioProcessor
from modules.text_processor import TextProcessor
from modules.fusion import CrossModalFusion
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TEST_CASES = [
    {"path": "./FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.mp4", "expected": 0.8, "label": "fake"},
    {"path": "./FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/women/id00220/00027_id00220_wavtolip.mp4", "expected": 0.75, "label": "fake"},
    {"path": "./FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00076/00109.mp4", "expected": 0.2, "label": "real"},
    {"path": "./FakeAVCeleb_v1.2/RealVideo-RealAudio/African/women/id00220/00027.mp4", "expected": 0.3, "label": "real"}
]

def sanitize_json(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def run_test_suite():
    """Run comprehensive test suite on fake and real videos"""
    results = []
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    fusion_module = CrossModalFusion()
    
    for test in TEST_CASES:
        print(f"\nTesting: {test['path']}")
        result = {
            "file": test['path'],
            "label": test['label'],
            "expected": test['expected']
        }
        
        # Run all processors
        modalities = {}
        try:
            # Video analysis
            start_time = time.time()
            modalities['video'] = video_processor.process(test['path'])
            result['video_time'] = time.time() - start_time
            result['video_confidence'] = modalities['video'].get('confidence', 0.5)
            
            # Audio analysis
            start_time = time.time()
            modalities['audio'] = audio_processor.process(test['path'])
            result['audio_time'] = time.time() - start_time
            result['audio_confidence'] = modalities['audio'].get('confidence', 0.5)
            
            # Text analysis
            start_time = time.time()
            modalities['text'] = text_processor.process(test['path'], None)
            result['text_time'] = time.time() - start_time
            result['text_confidence'] = modalities['text'].get('confidence', 0.5)
            
            # Fuse results
            start_time = time.time()
            fused = fusion_module.fuse_features(modalities)
            result['fusion_time'] = time.time() - start_time
            result['fused_confidence'] = fused.get('fused_confidence', 0.5)
            
            # Calculate accuracy
            result['is_deepfake'] = result['fused_confidence'] > 0.5
            result['correct'] = (result['is_deepfake'] and test['label'] == "fake") or \
                               (not result['is_deepfake'] and test['label'] == "real")
            
            print(f"  Fused confidence: {result['fused_confidence']:.2f} | Expected: {test['expected']:.2f} | Correct: {result['correct']}")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"  Error: {e}")
        
        results.append(result)
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(sanitize_json(results), f, indent=2)
    
    # Calculate metrics
    y_true = [1 if test['label'] == "fake" else 0 for test in TEST_CASES]
    y_pred = [1 if res.get('fused_confidence', 0.5) > 0.5 else 0 for res in results]
    
    if y_pred:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }
    else:
        metrics = {}
    
    # Print summary
    passed = sum(1 for r in results if r.get('correct', False))
    print(f"\nTest Summary: {passed}/{len(TEST_CASES)} passed")
    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.2f}")
    
    # Generate detailed report
    generate_test_report(results, metrics)
    
    return results, metrics

def generate_test_report(results, metrics):
    """Generate comprehensive test report"""
    report = {
        "timestamp": time.time(),
        "test_cases": results,
        "performance_metrics": metrics,
        "system_info": {
            "video_processor": "EfficientNet-B4",
            "audio_processor": "Whisper",
            "text_processor": "BERT+CLIP",
            "fusion_module": "CrossModalAttention"
        }
    }
    
    with open("test_report.json", "w") as f:
        json.dump(sanitize_json(report), f, indent=2)
    
    # Create CSV summary
    df = pd.DataFrame(results)
    df.to_csv("test_results_summary.csv", index=False)
    
    return report

if __name__ == "__main__":
    results, metrics = run_test_suite()
