"""
Evaluation utility functions for model testing and visualization.
"""

import os
from typing import Dict, List
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve


def print_evaluation_results(results: Dict, dataset_name: str, class_names: List[str] = None):
    """
    Print comprehensive evaluation results.
    
    Args:
        results: Dictionary containing evaluation metrics
        dataset_name: Name of the dataset being evaluated
        class_names: Optional list of class names for display
    """
    print(f"\n{'='*50}")
    print(f"{dataset_name} Evaluation Results")
    print(f"{'='*50}")
    
    print(f"Overall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  AU-ROC:    {results['auc_score']:.4f}")
    print(f"  Avg Loss:  {results['avg_loss']:.4f}")
    
    # Per-class metrics
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(results['per_class_precision']))]
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print(f"{'-'*50}")
    for i, class_name in enumerate(class_names):
        if i < len(results['per_class_precision']):
            print(f"{class_name:<15} {results['per_class_precision'][i]:<10.4f} "
                  f"{results['per_class_recall'][i]:<10.4f} {results['per_class_f1'][i]:<10.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"{'Actual \\ Predicted':<20}", end="")
    for i, class_name in enumerate(class_names):
        if i < cm.shape[1]:
            print(f"{class_name:<10}", end="")
    print()
    print("-" * (20 + 10 * cm.shape[1]))
    
    for i, class_name in enumerate(class_names):
        if i < cm.shape[0]:
            print(f"{class_name:<20}", end="")
            for j in range(cm.shape[1]):
                print(f"{cm[i, j]:<10}", end="")
            print()


def save_evaluation_plots(results: Dict, output_dir: str, dataset_name: str, class_names: List[str] = None):
    """
    Save evaluation plots including confusion matrix and ROC curves.
    
    Args:
        results: Dictionary containing evaluation metrics and data
        output_dir: Directory to save plots
        dataset_name: Name of the dataset for plot titles
        class_names: Optional list of class names for plot labels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(results['per_class_precision']))]
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{dataset_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve (for binary classification)
    if results['probabilities'].shape[1] == 2:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'][:, 1])
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {results["auc_score"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{dataset_name} - ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Evaluation plots saved to {output_dir}")


def save_evaluation_metrics(results: Dict, output_dir: str, dataset_name: str):
    """
    Save evaluation metrics to a JSON file.
    The file will be named "{dataset_name.lower()}_metrics.json" and placed in output_dir.

    Args:
        results: Dictionary containing evaluation metrics and data
        output_dir: Directory to save the metrics file
        dataset_name: Name of the dataset for the file name and payload
    """
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        'dataset': dataset_name,
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'auc_score': float(results['auc_score']),
        'avg_loss': float(results['avg_loss']),
        'per_class_precision': [float(x) for x in results['per_class_precision']],
        'per_class_recall': [float(x) for x in results['per_class_recall']],
        'per_class_f1': [float(x) for x in results['per_class_f1']],
        'confusion_matrix': results['confusion_matrix'].tolist()
    }

    out_path = os.path.join(output_dir, f"{dataset_name.lower()}_metrics.json")
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Evaluation metrics saved to {out_path}")
