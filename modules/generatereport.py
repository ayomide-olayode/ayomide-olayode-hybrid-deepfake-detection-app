import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def generate_chapter4_report():
    # Load collected data
    df = pd.read_csv('chapter4_results.csv')
    
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
    
    # Generate plots
    plt.figure(figsize=(15, 10))
    
    # Confusion Matrix Plot
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # ROC Curve Plot
    plt.subplot(2, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Confidence Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['final_confidence'], bins=20, kde=True)
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    
    # Modality Performance
    plt.subplot(2, 2, 4)
    modality_data = []
    for mod in ['video', 'audio', 'text']:
        if mod in df.columns:
            modality_data.append({
                'Modality': mod,
                'Accuracy': ((df[mod] > 0.5) == df['is_deepfake']).mean()
            })
    modality_df = pd.DataFrame(modality_data)
    sns.barplot(x='Modality', y='Accuracy', data=modality_df)
    plt.title('Modality Accuracy Comparison')
    
    plt.tight_layout()
    plt.savefig('chapter4_performance.png')
    
    return {
        "statistics": stats,
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
        "plots": "chapter4_performance.png"
    }