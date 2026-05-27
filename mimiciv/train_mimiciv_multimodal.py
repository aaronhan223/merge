"""
Train a multimodal TRUS MoE model on MIMIC-IV data.
Example usage:
python train_mimiciv_multimodal.py --train_data_path /path/to/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl --val_data_path /path/to/val_ihm-48-cxr-notes-missingInd-standardized_stays.pkl --rus_data_path /path/to/rus_multimodal_all_meanpool.npy --task ihm --use_wandb --gpu 0
python train_mimiciv_multimodal.py --train_data_path /path/to/train_los-cxr-notes-missingInd-standardized_stays.pkl --val_data_path /path/to/val_los-cxr-notes-missingInd-standardized_stays.pkl --rus_data_path /path/to/rus_multimodal_all_meanpool.npy --task los --use_wandb --gpu 0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
from typing import Dict, List, Tuple
from datetime import datetime
import random
import pickle
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import wandb
from sklearn.metrics import roc_auc_score
from mimiciv_rus_multimodal import preprocess_mimiciv_data
from model.trus_moe_multimodal import MultimodalTRUSMoEModel
from model.trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss
from plot_expert_activation import analyze_expert_activations


def load_mimiciv_rus_data(rus_filepath: str, modality_names: List[str], seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Loads MIMIC-IV RUS data with multiple lags and interpolates values across sequence length.
    """
    if not os.path.exists(rus_filepath):
        raise FileNotFoundError(f"RUS data file not found: {rus_filepath}")
    
    print(f"Loading RUS data from: {rus_filepath}")
    all_pid_results = np.load(rus_filepath, allow_pickle=True)
    num_modalities = len(modality_names)
    modality_to_idx = {name: idx for idx, name in enumerate(modality_names)}
    
    T = seq_len
    # Initialize modality-level tensors with time dimension
    U = torch.zeros(num_modalities, T, dtype=torch.float32)
    R = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    S = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    
    print(f"Expected modality names: {modality_names}")
    print(f"Total RUS results to process: {len(all_pid_results)}")
    print(f"Sequence length for interpolation: {T}")
    
    pairs_loaded = 0
    pairs_skipped_not_found = 0
    pairs_skipped_same_modality = 0
    
    # We'll keep track of processed modality pairs to avoid duplicates
    processed_pairs = set()
    
    for result in all_pid_results:
        mod1, mod2 = result['feature_pair']
        
        if mod1 not in modality_to_idx or mod2 not in modality_to_idx:
            print(f"Warning: Skipping pair ({mod1}, {mod2}) because one or both modalities not found.")
            pairs_skipped_not_found += 1
            continue
        
        m1_idx = modality_to_idx[mod1]
        m2_idx = modality_to_idx[mod2]

        if m1_idx == m2_idx:
            print(f"Warning: Skipping pair ({mod1}, {mod2}) because they are the same modality.")
            pairs_skipped_same_modality += 1
            continue
            
        # Create a key for the unordered pair
        pair_key = (min(m1_idx, m2_idx), max(m1_idx, m2_idx))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        lag_results = result['lag_results']
        num_lags = len(lag_results)
        
        # Extract lag values and RUS values
        lag_values = [lag_data['lag'] for lag_data in lag_results]
        R_values = [lag_data['R_value'] for lag_data in lag_results]
        S_values = [lag_data['S_value'] for lag_data in lag_results]
        U1_values = [lag_data['U1_value'] for lag_data in lag_results]
        U2_values = [lag_data['U2_value'] for lag_data in lag_results]
        
        print(f"Loading pair ({mod1}, {mod2}): {num_lags} lags at positions {lag_values}")
        
        # Map lag values directly to time indices (ensure within [0, T-1] range)
        lag_times = [min(lag, T - 1) for lag in lag_values]
        
        # Create full time range for interpolation
        full_times = torch.arange(T, dtype=torch.float32)
        
        # Helper function for interpolation
        def interpolate_values(values):
            if len(lag_times) == 1:
                return torch.full((T,), values[0])
            else:
                return torch.from_numpy(np.interp(full_times.numpy(), lag_times, values))
        
        # Interpolate R values
        R_interp = interpolate_values(R_values)
        R[m1_idx, m2_idx, :] = R_interp
        R[m2_idx, m1_idx, :] = R_interp  # Symmetric
        
        # Interpolate S values
        S_interp = interpolate_values(S_values)
        S[m1_idx, m2_idx, :] = S_interp
        S[m2_idx, m1_idx, :] = S_interp  # Symmetric
        
        # Interpolate U values for each modality (take max to handle overlapping pairs)
        U1_interp = interpolate_values(U1_values)
        U[m1_idx, :] = torch.maximum(U[m1_idx, :], U1_interp)
        
        U2_interp = interpolate_values(U2_values)
        U[m2_idx, :] = torch.maximum(U[m2_idx, :], U2_interp)
        pairs_loaded += 1
    
    print(f"Modality-level RUS data computed with interpolation. Shapes: U({U.shape}), R({R.shape}), S({S.shape})")
    print(f"  Average R value: {R.mean().item():.4f}")
    print(f"  Average S value: {S.mean().item():.4f}")
    print(f"  Average U value: {U.mean().item():.4f}")
    print(f"  Pairs loaded: {pairs_loaded}, Skipped (not found): {pairs_skipped_not_found}, Skipped (same): {pairs_skipped_same_modality}")

    return {'U': U, 'R': R, 'S': S}
        
class MultimodalMIMICIVDataset(Dataset):
    """
    PyTorch Dataset for multimodal MIMIC-IV data.
    """
    def __init__(self, multimodal_reg_ts: List[Dict[str, Tuple[np.ndarray, np.ndarray]]], labels: List[int], rus_data: Dict[str, torch.Tensor], modality_names: List[str], modality_dim_dict: Dict[str, int], max_seq_len: int = None, truncate_from_end: bool = True):
        # Convert all numpy arrays in multimodal_reg_ts to torch tensors
        # Sort modality keys to ensure consistent ordering across samples
        self.multimodal_reg_ts = []
        self.max_seq_len = max_seq_len
        self.truncate_from_end = truncate_from_end
        self.modality_names = sorted(modality_names)
        self.modality_dim_dict = modality_dim_dict
        # If max_seq_len is not provided, calculate it from the data
        if self.max_seq_len is None:
            max_len = 0
            seq_lengths = []
            for sample in multimodal_reg_ts:
                for modality, (features, mask) in sample.items():
                    seq_len = len(features)
                    max_len = max(max_len, seq_len)
                    seq_lengths.append(seq_len)
            self.max_seq_len = max_len
            print(f"Auto-detected max sequence length: {self.max_seq_len}")
            print(f"Sequence length statistics: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")
        else:
            print(f"Using specified max sequence length: {self.max_seq_len}")
            print(f"Truncation strategy: {'from end (keep first)' if self.truncate_from_end else 'from beginning (keep last)'}")
        
        # Initialize counters for statistics
        self.truncated_count = 0
        self.padded_count = 0
        
        for sample in multimodal_reg_ts:
            converted_sample = {}
            for modality in self.modality_names:
                if modality in sample:
                    features, mask = sample[modality]
                    # Convert numpy arrays to torch tensors
                    features_tensor = torch.from_numpy(features).float()
                    mask_tensor = torch.from_numpy(mask).bool()
                    # Apply truncation and padding
                    features_tensor, mask_tensor = self._truncate_and_pad(features_tensor, mask_tensor)
                else:
                    features_tensor, mask_tensor = torch.zeros(self.max_seq_len, self.modality_dim_dict[modality]), torch.zeros(self.max_seq_len, dtype=torch.bool)
                
                converted_sample[modality] = (features_tensor, mask_tensor)
            self.multimodal_reg_ts.append(converted_sample)
        
        # Convert numpy labels to Python integers to avoid tensor stacking issues
        self.labels = [int(label) for label in labels]
        self.rus_data = rus_data
        self.dataset_size = len(multimodal_reg_ts)
        
        # Print truncation/padding statistics
        if hasattr(self, 'truncated_count') and hasattr(self, 'padded_count'):
            print(f"Dataset preprocessing complete:")
            print(f"  Total samples: {self.dataset_size}")
            print(f"  Truncated sequences: {self.truncated_count}")
            print(f"  Padded sequences: {self.padded_count}")
            print(f"  No change needed: {self.dataset_size - self.truncated_count - self.padded_count}")

    def _truncate_and_pad(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Truncate or pad features and mask to match max_seq_len.
        
        Args:
            features: Tensor of shape (T, D) where T is current sequence length
            mask: Tensor of shape (T,) indicating valid timesteps
            
        Returns:
            Tuple of (padded_features, padded_mask) both with length max_seq_len
        """
        current_len = features.size(0)
        feature_dim = features.size(1)
        
        if current_len > self.max_seq_len:
            # Truncate based on truncate_from_end parameter
            if self.truncate_from_end:
                # Keep the first max_seq_len timesteps
                features = features[:self.max_seq_len]
                mask = mask[:self.max_seq_len]
            else:
                # Keep the last max_seq_len timesteps (most recent)
                features = features[-self.max_seq_len:]
                mask = mask[-self.max_seq_len:]
            self.truncated_count += 1
        elif current_len < self.max_seq_len:
            # Pad with zeros for features and False for mask
            pad_len = self.max_seq_len - current_len
            
            # Pad features with zeros
            feature_padding = torch.zeros(pad_len, feature_dim, dtype=features.dtype)
            features = torch.cat([features, feature_padding], dim=0)
            
            # Pad mask with False (indicating padded timesteps are invalid)
            mask_padding = torch.zeros(pad_len, dtype=mask.dtype)
            mask = torch.cat([mask, mask_padding], dim=0)
            self.padded_count += 1
        
        return features, mask

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.multimodal_reg_ts[idx], self.rus_data, self.labels[idx]


def collate_multimodal(batch):
    # Get sorted modality names from the first sample to ensure consistent ordering
    sorted_modality_names = sorted(batch[0][0].keys())
    
    modality_data_lists = [[] for _ in range(len(sorted_modality_names))]  # One list per modality
    modality_mask_lists = [[] for _ in range(len(sorted_modality_names))]  # One list per modality for masks
    rus_data_batch = {'U': [], 'R': [], 'S': []}
    labels = []
    
    for item in batch:
        modality_tensors, rus_dict, label = item
        
        for i, modality in enumerate(sorted_modality_names):
            features, mask = modality_tensors[modality]
            modality_data_lists[i].append(features)  # feature tensor
            modality_mask_lists[i].append(mask)      # mask tensor
        
        rus_data_batch['U'].append(rus_dict['U'])
        rus_data_batch['R'].append(rus_dict['R'])
        rus_data_batch['S'].append(rus_dict['S'])
        
        labels.append(label)
    
    # Stack data
    modality_batches = [torch.stack(mod_list) for mod_list in modality_data_lists]
    modality_mask_batches = [torch.stack(mask_list) for mask_list in modality_mask_lists]
    rus_batches = {k: torch.stack(v) for k, v in rus_data_batch.items()}
    label_batch = torch.tensor(labels, dtype=torch.long)
    return modality_batches, modality_mask_batches, rus_batches, label_batch


def train_epoch_multimodal_mimiciv(model: MultimodalTRUSMoEModel,
                                   dataloader: DataLoader,
                                   optimizer: optim.Optimizer,
                                   task_criterion: nn.Module,
                                   device: torch.device,
                                   args: argparse.Namespace,
                                   current_epoch: int):
    """Runs one training epoch for multimodal model."""
    # Start timing the epoch
    epoch_start_time = time.time()
    
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    unique_loss_accum = 0.0
    redundancy_loss_accum = 0.0
    synergy_loss_accum = 0.0
    load_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    all_scores = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Train]", leave=False)

    for batch_idx, (modality_data, modality_masks, rus_values_batch, labels) in enumerate(progress_bar):
        # Move data to device
        modality_data = [mod.to(device) for mod in modality_data]
        modality_masks = [mask.to(device) for mask in modality_masks]
        rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        final_logits, all_aux_moe_outputs = model(modality_data, rus_values)

        # Calculate Task Loss
        task_loss = task_criterion(final_logits, labels)

        # Calculate Auxiliary Losses
        total_L_unique = torch.tensor(0.0, device=device)
        total_L_redundancy = torch.tensor(0.0, device=device)
        total_L_synergy = torch.tensor(0.0, device=device)
        total_L_load = torch.tensor(0.0, device=device)
        num_moe_layers = len(all_aux_moe_outputs)
        
        for aux_outputs in all_aux_moe_outputs:
            gating_probs = aux_outputs['gating_probs']  # (B, M, T, N_exp)
            expert_indices = aux_outputs['expert_indices']  # (B, T, k)
            
            # Get synergy expert indices from the model
            
            # This is a bit tricky since we need to access the actual MoE layer
            # For now, we'll use a fixed set based on configuration
            num_experts = gating_probs.size(-1)
            synergy_expert_indices = set(range(args.moe_num_synergy_experts))
            
            # Calculate RUS losses
            L_unique, L_redundancy, L_synergy = calculate_rus_losses(
                gating_probs, rus_values, synergy_expert_indices,
                args.threshold_u, args.threshold_r, args.threshold_s,
                args.lambda_u, args.lambda_r, args.lambda_s,
                epsilon=args.epsilon_loss
            )
            

            # Calculate load balancing loss
            k = args.moe_k
            L_load = calculate_load_balancing_loss(gating_probs, expert_indices, k, args.lambda_load)

            total_L_unique += L_unique
            total_L_redundancy += L_redundancy
            total_L_synergy += L_synergy
            total_L_load += L_load

        if num_moe_layers > 0:
            total_L_unique /= num_moe_layers
            total_L_redundancy /= num_moe_layers
            total_L_synergy /= num_moe_layers
            total_L_load /= num_moe_layers

        # Combine Losses
        total_loss = task_loss + total_L_unique + total_L_redundancy + total_L_synergy + total_L_load

        # Backward pass and optimize
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN or Inf detected in total loss at batch {batch_idx}. Skipping.")
        else:
            total_loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
            
            # Accumulate losses and accuracy
            total_loss_accum += total_loss.item()
            task_loss_accum += task_loss.item()
            unique_loss_accum += total_L_unique.item()
            redundancy_loss_accum += total_L_redundancy.item()
            synergy_loss_accum += total_L_synergy.item()
            load_loss_accum += total_L_load.item()
            
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Store scores and labels for AU-ROC calculation
            probs = torch.softmax(final_logits, dim=1)[:, 1]  # Get probability of positive class
            all_scores.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if total_samples > 0:
                current_acc = 100. * correct_predictions / total_samples
                progress_bar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'TaskL': f"{task_loss.item():.4f}",
                    'Acc': f"{current_acc:.2f}%"
                })
    
    # Calculate average losses and accuracy
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0.0
    
    avg_total_loss = total_loss_accum / num_batches
    avg_task_loss = task_loss_accum / num_batches
    avg_unique_loss = unique_loss_accum / num_batches
    avg_redundancy_loss = redundancy_loss_accum / num_batches
    avg_synergy_loss = synergy_loss_accum / num_batches
    avg_load_loss = load_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # Calculate AU-ROC
    try:
        auc_score = roc_auc_score(all_labels, all_scores) if len(all_labels) > 0 else 0.0
    except ValueError:
        auc_score = 0.0  # In case all labels are the same class

    # Calculate epoch timing
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    # Log metrics to wandb
    if args.use_wandb:
        wandb.log({
            "train/total_loss": avg_total_loss,
            "train/task_loss": avg_task_loss,
            "train/unique_loss": avg_unique_loss,
            "train/redundancy_loss": avg_redundancy_loss,
            "train/synergy_loss": avg_synergy_loss,
            "train/load_balancing_loss": avg_load_loss,
            "train/accuracy": accuracy,
            "train/auc": auc_score,
            "train/epoch_duration_seconds": epoch_duration,
            "epoch": current_epoch + 1
        })
    
    print(f"Epoch {current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, "
          f"Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%, AU-ROC: {auc_score:.4f}")
    print(f"  Aux Losses -> Unique: {avg_unique_loss:.4f}, Redundancy: {avg_redundancy_loss:.4f}, "
          f"Synergy: {avg_synergy_loss:.4f}, Load: {avg_load_loss:.4f}")
    print(f"  Training Time: {epoch_duration:.2f}s")
    
    return avg_total_loss, auc_score, epoch_duration

def eval_epoch_multimodal_mimiciv(model: MultimodalTRUSMoEModel, dataloader: DataLoader, task_criterion: nn.Module, device: torch.device, args: argparse.Namespace, current_epoch: int):
    model.eval()
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    all_scores = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Val]", leave=False)
    
    with torch.no_grad():
        for batch_idx, (modality_data, modality_masks, rus_values_batch, labels) in enumerate(progress_bar):
            # Move data to device
            modality_data = [mod.to(device) for mod in modality_data]
            modality_masks = [mask.to(device) for mask in modality_masks]
            rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
            labels = labels.to(device)

            # Forward pass
            final_logits, _ = model(modality_data, rus_values)

            # Calculate Task Loss
            task_loss = task_criterion(final_logits, labels)
            task_loss_accum += task_loss.item()

            # Calculate accuracy
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Store scores and labels for AU-ROC calculation
            probs = torch.softmax(final_logits, dim=1)[:, 1]  # Get probability of positive class
            all_scores.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if total_samples > 0:
                current_acc = 100. * correct_predictions / total_samples
                progress_bar.set_postfix({
                    'Val TaskL': f"{task_loss.item():.4f}",
                    'Val Acc': f"{current_acc:.2f}%"
                })
    
    # Calculate average losses and accuracy
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0.0
    
    avg_task_loss = task_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # Calculate AU-ROC
    try:
        auc_score = roc_auc_score(all_labels, all_scores) if len(all_labels) > 0 else 0.0
    except ValueError:
        auc_score = 0.0  # In case all labels are the same class

    # Log validation metrics to wandb
    if args.use_wandb:
        wandb.log({
            "val/task_loss": avg_task_loss,
            "val/accuracy": accuracy,
            "val/auc": auc_score,
            "epoch": current_epoch + 1
        })
    
    print(f"Epoch {current_epoch+1} [Val] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%, AU-ROC: {auc_score:.4f}")
    
    return avg_task_loss, auc_score

def main(args):

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"mimiciv_multimodal_rus_moe_{args.task}_{timestamp}"

    # Set up wandb

    if args.use_wandb:        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=args.run_name,
            mode="online" if not args.wandb_disabled else "disabled"
        )


    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    modality_dim_dict = {'labs_vitals': 30,
                         'cxr': 1024,
                         'notes': 768}
    # Load data
    print(f"Loading train data from {args.train_data_path}...")
    train_stays = pickle.load(open(args.train_data_path, 'rb'))
    train_multimodal_reg_ts, train_labels = preprocess_mimiciv_data(train_stays, modality_dim_dict)
    print(f"Loading val data from {args.val_data_path}...")
    val_stays = pickle.load(open(args.val_data_path, 'rb'))
    val_multimodal_reg_ts, val_labels = preprocess_mimiciv_data(val_stays, modality_dim_dict)

    # Load RUS data
    modality_names = sorted(list(modality_dim_dict.keys()))
    num_classes = len(np.unique(train_labels))
    
    
    # Calculate baseline accuracy (majority class)
    train_baseline_acc = 100.0 * max(np.bincount(train_labels)) / len(train_labels)
    val_baseline_acc = 100.0 * max(np.bincount(val_labels)) / len(val_labels)
    print(f"\nBaseline Accuracy (Majority Class): Train={train_baseline_acc:.2f}%, Val={val_baseline_acc:.2f}%")
    
    rus_data = load_mimiciv_rus_data(args.rus_data_path, modality_names, args.seq_len)
    

    print(f"train multimodal reg ts: {train_multimodal_reg_ts[0].keys()}")

    # Create dataset
    train_dataset = MultimodalMIMICIVDataset(
        train_multimodal_reg_ts, train_labels, rus_data, modality_names, modality_dim_dict,
        max_seq_len=args.seq_len, truncate_from_end=args.truncate_from_end
    )

    val_dataset = MultimodalMIMICIVDataset(
        val_multimodal_reg_ts, val_labels, rus_data, modality_names, modality_dim_dict,
        max_seq_len=args.seq_len, truncate_from_end=args.truncate_from_end
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_multimodal
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_multimodal
    )

    # Create modality configs
    modality_configs = []
    # modality_dim_dict = dict()
    # for mod_name in modality_names:
    #     modality_dim_dict[mod_name] = train_multimodal_reg_ts[0][mod_name][0].shape[1]

    for mod_name in modality_names:
        config = {
            'input_dim': modality_dim_dict[mod_name],
            'num_layers': args.modality_encoder_layers,
            'nhead': args.nhead,
            'd_ff': args.d_ff,
            'use_cnn': args.use_cnn_encoders,
            'kernel_size': 3
        }
        modality_configs.append(config)

    # MoE configuration
    moe_router_config = {
        "gru_hidden_dim": args.moe_router_gru_hidden_dim,
        "token_processed_dim": args.moe_router_token_processed_dim,
        "attn_key_dim": args.moe_router_attn_key_dim,
        "attn_value_dim": args.moe_router_attn_value_dim,
    }
    
    moe_layer_config = {
        "num_experts": args.moe_num_experts,
        "num_synergy_experts": args.moe_num_synergy_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "synergy_expert_nhead": args.nhead,
        "router_config": moe_router_config,
        "capacity_factor": args.moe_capacity_factor,
        "drop_tokens": args.moe_drop_tokens,
    }

    print(f"Device: {device}")

    model = MultimodalTRUSMoEModel(
        modality_configs=modality_configs,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_layer_config,
        num_classes=num_classes,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        use_checkpoint=args.use_gradient_checkpointing,
        output_attention=False
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    

    
    os.makedirs(os.path.join(args.output_dir, args.task, 'checkpoints', args.run_name), exist_ok=True)

    task_criterion = nn.CrossEntropyLoss()
    best_val_auc = -1.0
    best_epoch = -1
    
    # Initialize list to track training times across epochs
    epoch_times = []
    
    # Reset peak memory counter once at the start of training
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(args.epochs):
        train_loss, train_auc, epoch_duration = train_epoch_multimodal_mimiciv(
            model, train_loader, optimizer, task_criterion, device, args, epoch
        )
        
        # Store timing metrics
        epoch_times.append(epoch_duration)

        val_loss, val_auc = eval_epoch_multimodal_mimiciv(
            model, val_loader, task_criterion, device, args, epoch
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            
            save_path = os.path.join(args.output_dir, args.task, 'checkpoints', args.run_name, f'best_multimodal_model_mimiciv.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'args': args,
                'modality_configs': modality_configs,
                'modality_names': modality_names,
            }, save_path)
            print(f"Epoch {epoch+1}: New best validation AU-ROC: {val_auc:.4f}. Model saved.")

            if args.use_wandb:
                wandb.log({"best_val_auc": best_val_auc, "best_epoch": epoch + 1})
            
        if scheduler is not None:
            scheduler.step()
            
    print("Training finished.")
    
    # Calculate and display training time and memory statistics
    if epoch_times:
        avg_epoch_time = np.mean(epoch_times)
        total_training_time = np.sum(epoch_times)
        print(f"\nTraining Time Statistics:")
        print(f"  Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"  Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")
        
        # Get peak GPU memory usage for entire training
        if device.type == "cuda":
            peak_memory_bytes = torch.cuda.max_memory_allocated(device)
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)  # Convert to MB
            print(f"  Peak GPU memory during training: {peak_memory_mb:.1f}MB")
            
            # Log summary statistics to wandb
            if args.use_wandb:
                wandb.log({
                    "summary/avg_epoch_time_seconds": avg_epoch_time,
                    "summary/total_training_time_seconds": total_training_time,
                    "summary/peak_memory_mb": peak_memory_mb
                })
        else:
            # Log summary statistics to wandb (CPU only)
            if args.use_wandb:
                wandb.log({
                    "summary/avg_epoch_time_seconds": avg_epoch_time,
                    "summary/total_training_time_seconds": total_training_time
                })
    
    if best_epoch != -1:
        print(f"Best Validation AU-ROC: {best_val_auc:.4f} at epoch {best_epoch+1}")

        if args.plot_expert_activations and len(val_loader) > 0:
            print("\nGenerating expert activation plots for the best multimodal TRUS-MoE model...")
            
            best_model_path = os.path.join(args.output_dir, args.task, 'checkpoints', args.run_name, f'best_multimodal_model_mimiciv.pth')
            if os.path.exists(best_model_path):
                # Add argparse.Namespace to safe globals for PyTorch 2.6+ compatibility
                torch.serialization.add_safe_globals([argparse.Namespace])
                checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                
                # Create a fresh model instance
                plot_model = MultimodalTRUSMoEModel(
                    modality_configs=modality_configs,
                    d_model=args.d_model,
                    nhead=args.nhead,
                    d_ff=args.d_ff,
                    num_encoder_layers=args.num_encoder_layers,
                    num_moe_layers=args.num_moe_layers,
                    moe_config=moe_layer_config,
                    num_classes=num_classes,
                    max_seq_len=args.seq_len,
                    dropout=args.dropout,
                    use_checkpoint=args.use_gradient_checkpointing,
                    output_attention=False
                ).to(device)

                plot_model.load_state_dict(checkpoint['model_state_dict'])
                plot_model.eval()
                
                # # Get a batch of validation data (manual approach - commented out)
                # val_batch_modalities = []
                # val_batch_rus = []
                # num_plot_samples = min(args.plot_num_samples, len(val_dataset))
                # 
                # for i in range(num_plot_samples):
                #     modality_data, rus_data, _ = val_dataset[i]
                #     val_batch_modalities.append(modality_data)
                #     val_batch_rus.append(rus_data)
                # 
                # # Stack into batch format
                # batch_modalities = [[] for _ in range(len(val_batch_modalities[0]))]
                # for sample_modalities in val_batch_modalities:
                #     for mod_idx, (mod_name, (features, _)) in enumerate(sorted(sample_modalities.items())):
                #         batch_modalities[mod_idx].append(features)
                # 
                # # Stack each modality
                # batch_modalities = [torch.stack(mod_list).to(device) for mod_list in batch_modalities]
                # 
                # # Stack RUS data
                # batch_rus = {'U': [], 'R': [], 'S': []}
                # for rus_data in val_batch_rus:
                #     batch_rus['U'].append(rus_data['U'])
                #     batch_rus['R'].append(rus_data['R'])
                #     batch_rus['S'].append(rus_data['S'])
                # batch_rus = {k: torch.stack(v).to(device) for k, v in batch_rus.items()}
                
                # Use validation dataloader for consistency (safer approach)
                print(f"Getting {args.plot_num_samples} samples from validation dataloader for expert activation plotting...")
                
                # Create a temporary dataloader with the desired batch size for plotting
                plot_loader = DataLoader(
                    val_dataset,
                    batch_size=args.plot_num_samples,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=collate_multimodal
                )
                
                plot_iter = iter(plot_loader)
                batch_modalities, batch_masks, batch_rus, batch_labels = next(plot_iter)
                # Move to device (same as training/validation loops)
                batch_modalities = [mod.to(device) for mod in batch_modalities]
                batch_rus = {k: v.to(device) for k, v in batch_rus.items()}
                
                # Generate plots
                plot_save_dir = os.path.join(args.output_dir, args.task, 'expert_activation_plots', args.run_name)
                
                try:
                    analyze_expert_activations(
                        time_moe_model=plot_model,
                        baseline_model=None,
                        data_batch=batch_modalities,
                        rus_values=batch_rus,
                        modality_names=modality_names,
                        save_dir=plot_save_dir
                    )
                    print(f"Expert activation plots saved to {plot_save_dir}")
                except Exception as e:
                    print(f"Error generating expert activation plots: {e}")
                    
            else:
                print(f"Best model checkpoint not found at {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multimodal TRUS-MoE Model on MIMIC-IV Data')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to the validation data')
    parser.add_argument('--rus_data_path', type=str, required=True, help='Path to the RUS data')
    parser.add_argument('--task', type=str, required=True, help='Task to train on')
    parser.add_argument('--seq_len', type=int, default=48, help='Sequence length (default 48 hours)')
    parser.add_argument('--truncate_from_end', action='store_true', 
                       help='Truncate sequences from the end (keep first timesteps). Default is to keep last timesteps.')
    # parser.add_argument('--rus_max_lag', type=int, default=10, help='Max lag used in RUS data')

    # Model architecture args
    parser.add_argument('--d_model', type=int, default=32, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=64, help='Feed-forward dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='Number of MoE layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--modality_encoder_layers', type=int, default=2, 
                       help='Number of layers in modality-specific encoders')
    parser.add_argument('--use_cnn_encoders', action='store_true', 
                       help='Use CNN layers in modality encoders')
    
    # MoE specific args
    parser.add_argument('--moe_num_experts', type=int, default=8, help='Number of experts per MoE layer')
    parser.add_argument('--moe_num_synergy_experts', type=int, default=2, help='Number of synergy experts')
    parser.add_argument('--moe_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128, help='Expert hidden dimension')
    parser.add_argument('--moe_capacity_factor', type=float, default=1.25, help='Expert capacity factor')
    parser.add_argument('--moe_drop_tokens', action='store_true', help='Drop tokens exceeding capacity')
    
    # MoE router args
    parser.add_argument('--moe_router_gru_hidden_dim', type=int, default=32, help='GRU hidden dim in router')
    parser.add_argument('--moe_router_token_processed_dim', type=int, default=32, 
                       help='Token processing dim in router')
    parser.add_argument('--moe_router_attn_key_dim', type=int, default=16, help='Attention key dim in router')
    parser.add_argument('--moe_router_attn_value_dim', type=int, default=16, 
                       help='Attention value dim in router')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, 
                       help='Max norm for gradient clipping (0 to disable)')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use cosine annealing LR scheduler')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', 
                       help='Use gradient checkpointing to save memory')
    
    # Loss args
    parser.add_argument('--threshold_u', type=float, default=0.01, help='Threshold for uniqueness loss')
    parser.add_argument('--threshold_r', type=float, default=0.01, help='Threshold for redundancy loss')
    parser.add_argument('--threshold_s', type=float, default=0.01, help='Threshold for synergy loss')
    parser.add_argument('--lambda_u', type=float, default=1, help='Weight for uniqueness loss')
    parser.add_argument('--lambda_r', type=float, default=1, help='Weight for redundancy loss')
    parser.add_argument('--lambda_s', type=float, default=1, help='Weight for synergy loss')
    parser.add_argument('--lambda_load', type=float, default=0.02, help='Weight for load balancing loss')
    parser.add_argument('--epsilon_loss', type=float, default=1e-8, help='Epsilon for loss stability')

    # System args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID to use. If None, will use CPUs.')
    parser.add_argument('--output_dir', type=str, default='./results', 
                       help='Directory to save results/models')
    
    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mimiciv-multimodal-trus-moe', 
                       help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
    parser.add_argument('--run_name', type=str, default=None, help='run name')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb')

    # Expert activation plotting args
    parser.add_argument('--plot_expert_activations', action='store_true', help='Generate expert activation plots after training')
    parser.add_argument('--plot_num_samples', type=int, default=32, help='Number of samples to use for expert activation plotting')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.task, 'checkpoints'), exist_ok=True)
    main(args)