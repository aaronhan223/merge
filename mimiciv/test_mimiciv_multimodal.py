import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
from typing import Dict
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path
# Import required modules
from mimiciv_rus_multimodal import preprocess_mimiciv_data
from model.trus_moe_multimodal import MultimodalTRUSMoEModel
from train_mimiciv_multimodal import (
    MultimodalMIMICIVDataset, 
    collate_multimodal, 
    load_mimiciv_rus_data
)
from utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint
from utils.evaluation_utils import print_evaluation_results, save_evaluation_plots, save_evaluation_metrics
from plot_expert_activation import analyze_expert_activations




def evaluate_model(model: MultimodalTRUSMoEModel, 
                  dataloader: DataLoader, 
                  device: torch.device,
                  dataset_name: str = "Test") -> Dict:
    """
    Evaluate model on given dataset and return comprehensive metrics.
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nEvaluating on {dataset_name} set...")
    progress_bar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
    
    with torch.no_grad():
        for batch_idx, (modality_data, modality_masks, rus_values_batch, labels) in enumerate(progress_bar):
            # Move data to device
            modality_data = [mod.to(device) for mod in modality_data]
            modality_masks = [mask.to(device) for mask in modality_masks]
            rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
            labels = labels.to(device)

            # Forward pass
            final_logits, _ = model(modality_data, rus_values)

            # Calculate loss
            loss = criterion(final_logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions and probabilities
            predictions = torch.argmax(final_logits, dim=1)
            probabilities = torch.softmax(final_logits, dim=1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = accuracy_score(all_labels, all_predictions)
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{current_acc:.4f}"
            })
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Basic metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # AU-ROC (for binary classification, use positive class probability)
    if all_probabilities.shape[1] == 2:
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
    else:
        # For multiclass, use one-vs-rest
        try:
            auc_score = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='weighted')
        except ValueError:
            auc_score = 0.0  # In case of issues with multiclass AUC
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Per-class metrics
    per_class_precision = precision_score(all_labels, all_predictions, average=None)
    per_class_recall = recall_score(all_labels, all_predictions, average=None)
    per_class_f1 = f1_score(all_labels, all_predictions, average=None)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'avg_loss': avg_loss,
        'confusion_matrix': cm,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels
    }
    
    return results




def main():
    parser = argparse.ArgumentParser(description='Test Multimodal TRUS-MoE Model on MIMIC-IV Data')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to the test data')
    parser.add_argument('--rus_data_path', type=str, required=True,
                       help='Path to the RUS data')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers for DataLoader')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use. If None, will use CPU.')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save evaluation plots')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--save_metrics', action='store_true',
                       help='Save evaluation metrics to JSON files in the checkpoint directory')
    
    # Additional evaluation options
    parser.add_argument('--eval_train', action='store_true',
                       help='Also evaluate on training set (requires --train_data_path)')
    parser.add_argument('--train_data_path', type=str, default=None,
                       help='Path to training data (for training set evaluation)')
    parser.add_argument('--eval_val', action='store_true',
                       help='Also evaluate on validation set (requires --val_data_path)')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='Path to validation data (for validation set evaluation)')
    
    # Expert activation plotting args
    parser.add_argument('--plot_expert_activations', action='store_true',
                       help='Generate expert activation plots after testing')
    parser.add_argument('--plot_num_samples', type=int, default=32,
                       help='Number of samples to use for expert activation plotting')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = Path(args.checkpoint_path).parent
    
    # Load checkpoint
    model_state_dict, train_args, modality_configs, modality_names, best_val_auc = load_checkpoint(
        args.checkpoint_path, device)
    
    modality_dim_dict = {'labs_vitals': 30, 'cxr': 1024, 'notes': 768}
    # Load test data
    print(f"Loading test data from {args.test_data_path}...")
    test_stays = pickle.load(open(args.test_data_path, 'rb'))
    test_multimodal_reg_ts, test_labels = preprocess_mimiciv_data(test_stays, modality_dim_dict)
    
    # Load RUS data
    print(f"Loading RUS data from {args.rus_data_path}...")
    rus_data = load_mimiciv_rus_data(args.rus_data_path, modality_names, train_args.seq_len)
    
    # Get data info
    num_classes = len(np.unique(test_labels))
    
    print(f"Test data info:")
    print(f"  Number of samples: {len(test_multimodal_reg_ts)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class distribution: {np.bincount(test_labels)}")
    
    # Create model
    model = create_model_from_checkpoint(model_state_dict, train_args, modality_configs, num_classes, device)
    
    # Create test dataset and dataloader
    test_dataset = MultimodalMIMICIVDataset(
        test_multimodal_reg_ts, test_labels, rus_data, modality_names, modality_dim_dict,
        max_seq_len=train_args.seq_len, truncate_from_end=train_args.truncate_from_end
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_multimodal
    )
    
    # Define class names (adjust based on your specific task)
    class_names = ['Negative', 'Positive']  # Adjust for your specific classification task
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, device, "Test")
    print_evaluation_results(test_results, "Test", class_names)
    if args.save_metrics:
        save_evaluation_metrics(test_results, str(args.output_dir), "Test")
    
    # Save test plots
    if args.save_plots:
        save_evaluation_plots(test_results, args.output_dir, "Test", class_names)
    
    # Save predictions
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, 'test_predictions.npz')
        np.savez(predictions_path,
                predictions=test_results['predictions'],
                probabilities=test_results['probabilities'],
                labels=test_results['labels'])
        print(f"Predictions saved to {predictions_path}")
    
    # Optional: Evaluate on training set
    if args.eval_train and args.train_data_path:
        print(f"\nLoading training data from {args.train_data_path}...")
        train_stays = pickle.load(open(args.train_data_path, 'rb'))
        train_multimodal_reg_ts, train_labels = preprocess_mimiciv_data(train_stays, modality_dim_dict)
        
        train_dataset = MultimodalMIMICIVDataset(
            train_multimodal_reg_ts, train_labels, rus_data, modality_names, modality_dim_dict,
            max_seq_len=train_args.seq_len, truncate_from_end=train_args.truncate_from_end
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_multimodal
        )
        
        train_results = evaluate_model(model, train_loader, device, "Train")
        print_evaluation_results(train_results, "Train", class_names)
        if args.save_metrics:
            save_evaluation_metrics(train_results, str(args.output_dir), "Train")
        
        if args.save_plots:
            save_evaluation_plots(train_results, args.output_dir, "Train", class_names)
    
    # Optional: Evaluate on validation set
    if args.eval_val and args.val_data_path:
        print(f"\nLoading validation data from {args.val_data_path}...")
        val_stays = pickle.load(open(args.val_data_path, 'rb'))
        val_multimodal_reg_ts, val_labels = preprocess_mimiciv_data(val_stays, modality_dim_dict)
        
        val_dataset = MultimodalMIMICIVDataset(
            val_multimodal_reg_ts, val_labels, rus_data, modality_names, modality_dim_dict,
            max_seq_len=train_args.seq_len, truncate_from_end=train_args.truncate_from_end
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_multimodal
        )
        
        val_results = evaluate_model(model, val_loader, device, "Validation")
        print_evaluation_results(val_results, "Validation", class_names)
        if args.save_metrics:
            save_evaluation_metrics(val_results, str(args.output_dir), "Validation")
        
        if args.save_plots:
            save_evaluation_plots(val_results, args.output_dir, "Validation", class_names)
    
    # Generate expert activation plots if requested
    if args.plot_expert_activations:
        print("\nGenerating expert activation plots for the test model...")
        
        # Create a temporary dataloader with the desired batch size for plotting
        plot_loader = DataLoader(
            test_dataset,
            batch_size=args.plot_num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_multimodal
        )
        
        plot_iter = iter(plot_loader)
        batch_modalities, batch_masks, batch_rus, batch_labels = next(plot_iter)
        # Move to device (same as evaluation loops)
        batch_modalities = [mod.to(device) for mod in batch_modalities]
        batch_rus = {k: v.to(device) for k, v in batch_rus.items()}
        
        # Generate plots
        plot_save_dir = os.path.join(args.output_dir, 'expert_activation_plots_test')
        
        try:
            analyze_expert_activations(
                time_moe_model=model,
                baseline_model=None,
                data_batch=batch_modalities,
                rus_values=batch_rus,
                modality_names=modality_names,
                save_dir=plot_save_dir,
                moe_num_synergy_experts=train_args.moe_num_synergy_experts
            )
            print(f"Expert activation plots saved to {plot_save_dir}")
        except Exception as e:
            print(f"Error generating expert activation plots: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Best training validation AU-ROC: {best_val_auc:.4f}")
    print(f"Test AU-ROC: {test_results['auc_score']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
