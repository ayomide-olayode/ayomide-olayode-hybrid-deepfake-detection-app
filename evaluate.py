import os
import pandas as pd
from tqdm import tqdm
from modules.video_processor import VideoProcessor
from modules.audio_processor import AudioProcessor
from modules.text_processor import TextProcessor
from modules.fusion import CrossModalFusion
from modules.metrics import MetricsCalculator

def evaluate_dataset(dataset_path, max_samples_per_category=50):
    # Load metadata
    metadata_path = os.path.join(dataset_path, 'meta_data.csv')
    metadata = pd.read_csv(metadata_path)
    
    # Initialize processors
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    fusion = CrossModalFusion()
    
    # Initialize metrics
    metrics = MetricsCalculator()
    results = []
    
    # Process each category
    categories = {
        'A': {'label': 0, 'type': 'real'},
        'B': {'label': 1, 'type': 'fake_audio'},
        'C': {'label': 1, 'type': 'fake_video'},
        'D': {'label': 1, 'type': 'fake_both'}
    }
    
    for cat, cat_info in categories.items():
        cat_files = metadata[metadata['category'] == cat]
        if len(cat_files) > max_samples_per_category:
            cat_files = cat_files.sample(max_samples_per_category)
        
        for _, row in tqdm(cat_files.iterrows(), desc=f"Processing {cat}"):
            video_path = os.path.join(dataset_path, row['category'], row['path'], row['filename'])
            
            try:
                # Process through full pipeline
                video_result = video_processor.process(video_path)
                audio_result = audio_processor.process(video_path)
                text_result = text_processor.process(video_path, audio_result['transcription']['text'])
                
                # Fuse results
                fused = fusion.fuse_features({
                    'video': video_result,
                    'audio': audio_result,
                    'text': text_result
                })
                
                # Add to metrics
                metrics.add_sample(
                    ground_truth=cat_info['label'],
                    prediction=1 if fused['fused_confidence'] > 0.5 else 0,
                    confidence=fused['fused_confidence']
                )
                
                # Store detailed results
                results.append({
                    'file': row['filename'],
                    'category': cat,
                    'type': cat_info['type'],
                    'method': row.get('method', 'N/A'),
                    'race': row['race'],
                    'gender': row['gender'],
                    'video_confidence': video_result['confidence'],
                    'audio_confidence': audio_result['confidence'],
                    'text_confidence': text_result['confidence'],
                    'fused_confidence': fused['fused_confidence'],
                    'label': cat_info['label']
                })
                
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
    
    # Calculate and return metrics
    metrics_report = metrics.calculate_all_metrics()
    results_df = pd.DataFrame(results)
    
    return {
        'overall_metrics': metrics_report,
        'detailed_results': results_df,
        'category_metrics': calculate_category_metrics(results_df),
        'method_metrics': calculate_method_metrics(results_df)
    }

def calculate_category_metrics(results_df):
    metrics = {}
    for category in results_df['category'].unique():
        cat_data = results_df[results_df['category'] == category]
        y_true = cat_data['label']
        y_score = cat_data['fused_confidence']
        metrics[category] = calculate_metrics(y_true, y_score)
    return metrics

def calculate_method_metrics(results_df):
    metrics = {}
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        if len(method_data) > 0:
            y_true = method_data['label']
            y_score = method_data['fused_confidence']
            metrics[method] = calculate_metrics(y_true, y_score)
    return metrics

def calculate_metrics(y_true, y_score):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred = [1 if score > 0.5 else 0 for score in y_score]
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_score)
    }

# Run evaluation
if __name__ == "__main__":
    dataset_path = "./FakeAVCeleb_v1.2"
    evaluation_results = evaluate_dataset(dataset_path)
    
    # Save results
    evaluation_results['detailed_results'].to_csv('evaluation_results.csv', index=False)
    
    # Print summary
    print("\n===== Overall Metrics =====")
    print(f"Accuracy: {evaluation_results['overall_metrics']['accuracy']:.4f}")
    print(f"F1 Score: {evaluation_results['overall_metrics']['f1_score']:.4f}")
    print(f"ROC AUC: {evaluation_results['overall_metrics']['roc_auc']:.4f}")
    
    print("\n===== Category Metrics =====")
    for cat, metrics in evaluation_results['category_metrics'].items():
        print(f"\nCategory {cat}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    print("\n===== Method Metrics =====")
    for method, metrics in evaluation_results['method_metrics'].items():
        print(f"\nMethod {method}:")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")