import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import math
import os
import sys
import argparse
import random
import wandb
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Set
import pdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model.trus_moe_multimodal import MultimodalTRUSMoEModel
    from model.trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss
    from pamap_rus import get_pamap_column_names, load_pamap_data, preprocess_pamap_data
    from plot_expert_activation import analyze_expert_activations
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Please ensure trus_moe_multimodal.py, trus_moe_model.py, pamap_rus.py, and plot_expert_activation.py are accessible.")
    sys.exit(1)


DEFAULT_DATASET_DIR = "/cis/home/xhan56/pamap/PAMAP2_Dataset/Protocol"
DEFAULT_OUTPUT_DIR = "./results/pamap_multimodal_training"
DEFAULT_RUS_FILE_PATTERN = "./results/pamap/pamap_subject{SUBJECT_ID}_lag{MAX_LAG}_{METHOD}_discrim_epochs{DISCRIM_EPOCHS}_ce_epochs{CE_EPOCHS}_n_batches{N_BATCHES}_batch_size{BATCH_SIZE}_seed{SEED}.npy"


def parse_subject_list(s):
    """Parses '1', '1,2,3', or '1-6' into a list of subject ids."""
    s = str(s).strip()
    if '-' in s and ',' not in s:
        start, end = s.split('-')
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(',') if x.strip()]


def format_subject_tag(subject_ids):
    """Compact tag for filenames, e.g. [1,2,3,4,5,6] -> '1-6'."""
    if not subject_ids:
        return "none"
    if len(subject_ids) == 1:
        return str(subject_ids[0])
    sorted_ids = sorted(subject_ids)
    if sorted_ids == list(range(sorted_ids[0], sorted_ids[-1] + 1)):
        return f"{sorted_ids[0]}-{sorted_ids[-1]}"
    return ",".join(str(s) for s in sorted_ids)


def categorize_pamap_sensors(sensor_columns: List[str]) -> Dict[str, List[str]]:
    """
    Categorizes PAMAP sensor columns by body location.
    
    Expected sensor naming convention:
    - {sensor_type}_{location}_{axis}
    - e.g., acc_chest_x, gyro_hand_y, mag_ankle_z
    
    Returns:
        Dictionary mapping modality names to lists of sensor columns
    """
    modality_sensors = {
        'chest': [],
        'hand': [],
        'ankle': [],
        'heart_rate': []
    }
    
    for col in sensor_columns:
        col_lower = col.lower()
        
        # Check for heart rate
        if 'heart' in col_lower or 'hr' in col_lower:
            modality_sensors['heart_rate'].append(col)
        # Check for body locations
        elif 'chest' in col_lower:
            modality_sensors['chest'].append(col)
        elif 'hand' in col_lower:
            modality_sensors['hand'].append(col)
        elif 'ankle' in col_lower:
            modality_sensors['ankle'].append(col)
        else:
            # Try to infer from other patterns
            # Sometimes sensors might be named differently
            print(f"Warning: Could not categorize sensor column: {col}")
    
    # Remove empty modalities
    modality_sensors = {k: v for k, v in modality_sensors.items() if v}
    
    return modality_sensors


def load_multimodal_rus_data(rus_filepath: str, modality_sensors: Dict[str, List[str]], 
                            seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Loads RUS data and computes modality-level RUS values.
    
    For multimodal setting, we need to process modality-level RUS values directly.
    """
    if not os.path.exists(rus_filepath):
        raise FileNotFoundError(f"RUS data file not found: {rus_filepath}")
    
    print(f"Loading RUS data from: {rus_filepath}")
    all_pid_results = np.load(rus_filepath, allow_pickle=True)
    
    # Get modality names and create mapping to index
    modality_names = list(modality_sensors.keys())
    num_modalities = len(modality_names)
    modality_to_idx = {name: idx for idx, name in enumerate(modality_names)}
    
    T = seq_len
    # Initialize modality-level tensors
    U = torch.zeros(num_modalities, T, dtype=torch.float32)
    R = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    S = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    
    # We'll keep track of processed modality pairs to avoid duplicates
    processed_pairs = set()
    
    for result in all_pid_results:
        # The feature_pair now contains two modality names
        mod1, mod2 = result['feature_pair']
        
        # Skip if either modality is not in our list
        if mod1 not in modality_to_idx or mod2 not in modality_to_idx:
            print(f"Warning: Skipping pair ({mod1}, {mod2}) because one or both modalities not found.")
            continue
            
        m1_idx = modality_to_idx[mod1]
        m2_idx = modality_to_idx[mod2]
        
        # Skip if same modality (shouldn't happen, but just in case)
        if m1_idx == m2_idx:
            continue
            
        # Create a key for the unordered pair
        pair_key = (min(m1_idx, m2_idx), max(m1_idx, m2_idx))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        lag_results = result['lag_results']
        num_lags = len(lag_results)
        segment_length = max(1, T // num_lags)
        
        for lag_idx, lag_data in enumerate(lag_results):
            start_idx = lag_idx * segment_length
            end_idx = min(T, (lag_idx + 1) * segment_length)
            
            if lag_idx == num_lags - 1:
                end_idx = T
                
            if start_idx >= T:
                break
                
            # Get the R, S, and U values for this lag
            R_value = lag_data['R_value']
            S_value = lag_data['S_value']
            U1_value = lag_data['U1_value']
            U2_value = lag_data['U2_value']

            # Update R and S for the pair (symmetric)
            R[m1_idx, m2_idx, start_idx:end_idx] = torch.tensor(R_value)
            R[m2_idx, m1_idx, start_idx:end_idx] = torch.tensor(R_value)
            
            S[m1_idx, m2_idx, start_idx:end_idx] = torch.tensor(S_value)
            S[m2_idx, m1_idx, start_idx:end_idx] = torch.tensor(S_value)
            
            # Update U for each modality: take the max of the current segment and the new U value
            # For modality m1_idx
            current_segment_U1 = U[m1_idx, start_idx:end_idx]
            current_max_U1 = current_segment_U1.max().item()
            new_value_U1 = max(current_max_U1, U1_value)
            U[m1_idx, start_idx:end_idx] = torch.tensor(new_value_U1)
            
            # For modality m2_idx
            current_segment_U2 = U[m2_idx, start_idx:end_idx]
            current_max_U2 = current_segment_U2.max().item()
            new_value_U2 = max(current_max_U2, U2_value)
            U[m2_idx, start_idx:end_idx] = torch.tensor(new_value_U2)

    print(f"Modality-level RUS data computed. Shapes: U({U.shape}), R({R.shape}), S({S.shape})")
    print(f"  Average R value: {R.mean().item():.4f}")
    print(f"  Average S value: {S.mean().item():.4f}")
    print(f"  Average U value: {U.mean().item():.4f}")

    return {'U': U, 'R': R, 'S': S}


class MultimodalPamapDataset(Dataset):
    """
    PyTorch Dataset for multimodal PAMAP2 activity recognition over one or more subjects.
    Each window is paired with the RUS values of the subject it came from, so a single
    Dataset can hold a full train/val/test split without leaking RUS information across
    subjects.
    """
    def __init__(self, subject_ids: List[int], data_dir: str,
                 rus_data_per_subject: Dict[int, Dict[str, torch.Tensor]],
                 modality_sensors: Dict[str, List[str]], seq_len: int, step: int,
                 activity_map: Dict[int, int]):
        """
        Args:
            subject_ids: List of subject IDs included in this split.
            data_dir: Directory containing PAMAP2 .dat files.
            rus_data_per_subject: Maps subject_id -> {'U','R','S'} tensors for that subject.
            modality_sensors: Dictionary mapping modality names to sensor columns.
            seq_len: Length of the sliding window (T).
            step: Step size for the sliding window.
            activity_map: Dictionary mapping original activity IDs to 0-based indices.
        """
        self.subject_ids = list(subject_ids)
        self.data_dir = data_dir
        self.rus_data_per_subject = rus_data_per_subject
        self.modality_sensors = modality_sensors
        self.modality_names = list(modality_sensors.keys())
        self.num_modalities = len(self.modality_names)
        self.seq_len = seq_len
        self.step = step
        self.activity_map = activity_map

        # Get all sensor columns in order
        all_sensor_columns = []
        for sensors in modality_sensors.values():
            all_sensor_columns.extend(sensors)
        self._all_sensor_columns = all_sensor_columns

        self.windows: List[List[torch.Tensor]] = []
        self.rus_refs: List[Dict[str, torch.Tensor]] = []
        self.labels: List[torch.Tensor] = []

        for sid in self.subject_ids:
            if sid not in self.rus_data_per_subject:
                raise KeyError(f"RUS data missing for subject {sid}")
            self._load_subject(sid)

    def _load_subject(self, subject_id: int):
        """Loads and windows a single subject, appending to the dataset state."""
        try:
            df = load_pamap_data(subject_id, self.data_dir)
        except Exception as e:
            print(f"Error loading subject {subject_id}: {e}")
            raise

        selected_cols_with_id = ['timestamp', 'activity_id'] + self._all_sensor_columns
        cols_to_use = [col for col in selected_cols_with_id if col in df.columns]
        missing_cols = set(selected_cols_with_id) - set(cols_to_use)
        if missing_cols:
            print(f"Warning [subject {subject_id}]: missing sensor columns: {missing_cols}")

        df_subset = df[cols_to_use].copy()
        df_processed, _ = preprocess_pamap_data(df_subset)

        df_processed['activity_label'] = df_processed['activity_id'].map(self.activity_map)
        df_processed.dropna(subset=['activity_label'], inplace=True)
        df_processed['activity_label'] = df_processed['activity_label'].astype(int)

        if df_processed.empty:
            print(f"Warning: No data remaining for subject {subject_id}")
            return

        # Per-modality numpy views for sliding-window extraction
        modality_data = {}
        for mod_name, sensors in self.modality_sensors.items():
            existing_sensors = [s for s in sensors if s in df_processed.columns]
            if existing_sensors:
                modality_data[mod_name] = df_processed[existing_sensors].values
            else:
                print(f"Warning [subject {subject_id}]: no sensors for modality {mod_name}; zero-filling")
                modality_data[mod_name] = np.zeros((len(df_processed), 1))

        label_values = df_processed['activity_label'].values
        total_samples = len(df_processed)
        subject_rus = self.rus_data_per_subject[subject_id]

        n_added = 0
        for i in range(0, total_samples - self.seq_len + 1, self.step):
            window_data_by_modality = [
                torch.tensor(modality_data[mod_name][i:i + self.seq_len], dtype=torch.float32)
                for mod_name in self.modality_names
            ]
            window_labels = label_values[i:i + self.seq_len]
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            most_frequent_label = unique_labels[np.argmax(counts)]

            self.windows.append(window_data_by_modality)
            self.rus_refs.append(subject_rus)
            self.labels.append(torch.tensor(most_frequent_label, dtype=torch.long))
            n_added += 1

        print(f"Subject {subject_id}: created {n_added} windows.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Returns: list of modality tensors, RUS dict (subject-specific), label
        return self.windows[idx], self.rus_refs[idx], self.labels[idx]


def collate_multimodal(batch):
    """
    Custom collate function for multimodal data.
    """
    modality_data_lists = [[] for _ in range(len(batch[0][0]))]  # One list per modality
    rus_data_batch = {'U': [], 'R': [], 'S': []}
    labels = []
    
    for item in batch:
        modality_tensors, rus_dict, label = item
        
        # Collect modality data
        for i, mod_tensor in enumerate(modality_tensors):
            modality_data_lists[i].append(mod_tensor)
        
        # Collect RUS data
        rus_data_batch['U'].append(rus_dict['U'])
        rus_data_batch['R'].append(rus_dict['R'])
        rus_data_batch['S'].append(rus_dict['S'])
        
        # Collect labels
        labels.append(label)
    
    # Stack data
    modality_batches = [torch.stack(mod_list) for mod_list in modality_data_lists]
    rus_batches = {k: torch.stack(v) for k, v in rus_data_batch.items()}
    label_batch = torch.stack(labels)
    
    return modality_batches, rus_batches, label_batch


def train_epoch_multimodal(model: MultimodalTRUSMoEModel,
                          dataloader: DataLoader,
                          optimizer: optim.Optimizer,
                          task_criterion: nn.Module,
                          device: torch.device,
                          args: argparse.Namespace,
                          current_epoch: int):
    """Runs one training epoch for multimodal model."""
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    unique_loss_accum = 0.0
    redundancy_loss_accum = 0.0
    synergy_loss_accum = 0.0
    load_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Set epoch for distributed sampler if using
    if args.distributed and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(current_epoch)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Train]", 
                        leave=False, disable=args.distributed and dist.get_rank() != 0)
    
    for batch_idx, (modality_data, rus_values_batch, labels) in enumerate(progress_bar):
        # Move data to device
        modality_data = [mod.to(device) for mod in modality_data]
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
    
    # Log metrics to wandb
    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        wandb.log({
            "train/total_loss": avg_total_loss,
            "train/task_loss": avg_task_loss,
            "train/unique_loss": avg_unique_loss,
            "train/redundancy_loss": avg_redundancy_loss,
            "train/synergy_loss": avg_synergy_loss,
            "train/load_balancing_loss": avg_load_loss,
            "train/accuracy": accuracy,
            "epoch": current_epoch + 1
        })
    
    print(f"Epoch {current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, "
          f"Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"  Aux Losses -> Unique: {avg_unique_loss:.4f}, Redundancy: {avg_redundancy_loss:.4f}, "
          f"Synergy: {avg_synergy_loss:.4f}, Load: {avg_load_loss:.4f}")
    
    return avg_total_loss, accuracy


def evaluate_multimodal(model: MultimodalTRUSMoEModel,
                        dataloader: DataLoader,
                        task_criterion: nn.Module,
                        device: torch.device,
                        args: argparse.Namespace,
                        split_name: str = "Val",
                        current_epoch: Optional[int] = None,
                        wandb_prefix: Optional[str] = None):
    """Runs one evaluation pass for the multimodal model on the provided dataloader."""
    model.eval()
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0

    desc = f"[{split_name}]" if current_epoch is None else f"Epoch {current_epoch+1}/{args.epochs} [{split_name}]"
    progress_bar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch_idx, (modality_data, rus_values_batch, labels) in enumerate(progress_bar):
            modality_data = [mod.to(device) for mod in modality_data]
            rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
            labels = labels.to(device)

            final_logits, _ = model(modality_data, rus_values)
            task_loss = task_criterion(final_logits, labels)
            task_loss_accum += task_loss.item()

            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            if total_samples > 0:
                current_acc = 100. * correct_predictions / total_samples
                progress_bar.set_postfix({
                    f'{split_name} TaskL': f"{task_loss.item():.4f}",
                    f'{split_name} Acc': f"{current_acc:.2f}%"
                })

    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0.0

    avg_task_loss = task_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0

    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        prefix = wandb_prefix if wandb_prefix is not None else split_name.lower()
        log_dict = {f"{prefix}/task_loss": avg_task_loss, f"{prefix}/accuracy": accuracy}
        if current_epoch is not None:
            log_dict["epoch"] = current_epoch + 1
        wandb.log(log_dict)

    epoch_str = "" if current_epoch is None else f"Epoch {current_epoch+1} "
    print(f"{epoch_str}[{split_name}] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_task_loss, accuracy


# Backwards-compatible alias used inside the training loop.
def validate_epoch_multimodal(model, dataloader, task_criterion, device, args, current_epoch):
    return evaluate_multimodal(
        model, dataloader, task_criterion, device, args,
        split_name="Val", current_epoch=current_epoch, wandb_prefix="val",
    )


def main(args):
    """Main function to set up and run multimodal training."""
    # Parse subject splits up front so they're available for naming and bookkeeping
    train_subjects = parse_subject_list(args.train_subjects)
    val_subjects = parse_subject_list(args.val_subjects)
    test_subjects = parse_subject_list(args.test_subjects)
    all_subjects = list(dict.fromkeys(train_subjects + val_subjects + test_subjects))
    split_tag = (
        f"train{format_subject_tag(train_subjects)}"
        f"_val{format_subject_tag(val_subjects)}"
        f"_test{format_subject_tag(test_subjects)}"
    )

    # Initialize wandb if enabled
    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        wandb_config = {k: v for k, v in vars(args).items()}
        run_name = f"multimodal_{split_tag}_seq{args.seq_len}"
        if args.wandb_run_name:
            run_name = args.wandb_run_name

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_config,
            name=run_name,
            mode="online" if not args.wandb_disabled else "disabled"
        )
    
    # Set up distributed training if requested
    if args.distributed:
        if 'LOCAL_RANK' not in os.environ:
            raise ValueError("For distributed training, please launch with torchrun")
        
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ.get('RANK', local_rank))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl')
        is_main_process = global_rank == 0
        
        if is_main_process:
            print(f"Distributed training initialized with world_size: {world_size}")
    else:
        # Non-distributed mode
        if args.cuda_device >= 0 and torch.cuda.is_available() and not args.no_cuda:
            device = torch.device(f"cuda:{args.cuda_device}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        is_main_process = True
    
    if is_main_process:
        print(f"Using device: {device}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load initial data per subject to (a) build a shared sensor/modality view and
    # (b) derive a global activity map that covers every split.
    print(f"Loading data for subjects {all_subjects} to derive sensors and activity map...")
    modality_sensors = None
    unique_activities_set = set()
    try:
        for sid in all_subjects:
            temp_df = load_pamap_data(sid, args.data_dir)
            temp_df_processed, all_sensor_columns = preprocess_pamap_data(temp_df)

            sid_modality_sensors = categorize_pamap_sensors(all_sensor_columns)
            if not sid_modality_sensors:
                print(f"Error: No sensors could be categorized for subject {sid}.")
                sys.exit(1)

            if modality_sensors is None:
                modality_sensors = sid_modality_sensors
            elif list(modality_sensors.keys()) != list(sid_modality_sensors.keys()):
                print(f"Warning: modality set differs for subject {sid} "
                      f"({list(sid_modality_sensors.keys())} vs "
                      f"{list(modality_sensors.keys())}). Using first-seen layout.")

            unique_activities_set.update(
                int(a) for a in temp_df_processed['activity_id'].unique() if a != 0
            )

        unique_activities = sorted(unique_activities_set)
        activity_map = {activity_id: i for i, activity_id in enumerate(unique_activities)}
        num_classes = len(activity_map)

        print(f"Found {len(modality_sensors)} modalities:")
        for mod_name, sensors in modality_sensors.items():
            print(f"  {mod_name}: {len(sensors)} sensors")
        print(f"Global activity map ({num_classes} classes): {unique_activities}")

    except Exception as e:
        print(f"Error during initial data loading: {e}")
        sys.exit(1)

    # Load per-subject RUS data. Each split uses only the RUS values of its own
    # subjects, so RUS for test subjects is held out until inference.
    rus_data_per_subject = {}
    rus_file_names = {}
    for sid in all_subjects:
        rus_file = args.rus_file_pattern.format(
            SUBJECT_ID=sid,
            MAX_LAG=args.rus_max_lag,
            METHOD=args.rus_method,
            DISCRIM_EPOCHS=args.rus_discrim_epochs,
            CE_EPOCHS=args.rus_ce_epochs,
            N_BATCHES=args.rus_n_batches,
            BATCH_SIZE=args.rus_batch_size,
            SEED=args.rus_seed,
        )
        rus_file_names[sid] = os.path.splitext(os.path.basename(rus_file))[0]
        rus_data_per_subject[sid] = load_multimodal_rus_data(rus_file, modality_sensors, args.seq_len)

    # A single stable tag for save filenames (mirrors the RUS hyperparameters)
    rus_hparams_tag = (
        f"lag{args.rus_max_lag}_{args.rus_method}"
        f"_discrim_epochs{args.rus_discrim_epochs}_ce_epochs{args.rus_ce_epochs}"
        f"_n_batches{args.rus_n_batches}_batch_size{args.rus_batch_size}_seed{args.rus_seed}"
    )
    run_tag = f"pamap_{split_tag}_{rus_hparams_tag}"

    if is_main_process:
        print("Creating multimodal datasets for train / val / test splits...")

    def build_split(subject_subset, split_label):
        if not subject_subset:
            return None
        rus_subset = {sid: rus_data_per_subject[sid] for sid in subject_subset}
        ds = MultimodalPamapDataset(
            subject_ids=subject_subset,
            data_dir=args.data_dir,
            rus_data_per_subject=rus_subset,
            modality_sensors=modality_sensors,
            seq_len=args.seq_len,
            step=args.window_step,
            activity_map=activity_map,
        )
        print(f"{split_label} dataset size: {len(ds)} windows from subjects {subject_subset}")
        return ds

    try:
        train_dataset = build_split(train_subjects, "Train")
        val_dataset = build_split(val_subjects, "Val")
        test_dataset = build_split(test_subjects, "Test")
    except Exception as e:
        if is_main_process:
            print(f"Error creating datasets: {e}")
        if args.distributed:
            dist.destroy_process_group()
        sys.exit(1)

    if train_dataset is None or len(train_dataset) == 0:
        if is_main_process:
            print("Error: Training dataset is empty.")
        if args.distributed:
            dist.destroy_process_group()
        sys.exit(1)

    val_size = len(val_dataset) if val_dataset is not None else 0
    test_size = len(test_dataset) if test_dataset is not None else 0

    # Create samplers for distributed training
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                          rank=dist.get_rank(), shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(),
                                        rank=dist.get_rank(), shuffle=False, seed=args.seed) if val_size > 0 else None
        test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                         rank=dist.get_rank(), shuffle=False, seed=args.seed) if test_size > 0 else None
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal
    ) if val_size > 0 else []

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal
    ) if test_size > 0 else []
    
    # Model configuration
    modality_names = list(modality_sensors.keys())
    modality_configs = []
    
    for mod_name in modality_names:
        num_sensors = len(modality_sensors[mod_name])
        config = {
            'input_dim': num_sensors,
            'num_layers': args.modality_encoder_layers,
            'nhead': args.nhead,
            'd_ff': args.d_ff,
            'use_cnn': args.use_cnn_encoders and mod_name != 'heart_rate',  # No CNN for heart rate
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
    
    if is_main_process:
        print("Initializing multimodal TRUS-MoE model...")
    
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
    
    # Wrap model with DDP if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                   find_unused_parameters=False)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    task_criterion = nn.CrossEntropyLoss()
    
    # Training loop
    if is_main_process:
        print("Starting training...")
    
    best_val_accuracy = -1.0
    best_epoch = -1
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_multimodal(
            model, train_loader, optimizer, task_criterion, device, args, epoch
        )
        
        if len(val_loader) > 0:
            val_loss, val_acc = validate_epoch_multimodal(
                model, val_loader, task_criterion, device, args, epoch
            )
            
            # For distributed training, all processes should have same val_acc
            if args.distributed:
                val_acc_tensor = torch.tensor([val_acc], device=device)
                dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
                val_acc = val_acc_tensor.item() / dist.get_world_size()
            
            # Save best model
            if val_acc > best_val_accuracy and is_main_process:
                best_val_accuracy = val_acc
                best_epoch = epoch

                save_path = os.path.join(args.output_dir, f'best_model_{run_tag}.pth')

                # Save model state dict without DDP wrapper if distributed
                if args.distributed:
                    model_to_save = model.module
                else:
                    model_to_save = model

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy,
                    'args': args,
                    'modality_configs': modality_configs,
                    'modality_names': modality_names,
                    'train_subjects': train_subjects,
                    'val_subjects': val_subjects,
                    'test_subjects': test_subjects,
                }, save_path)

                print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%. Model saved.")

                # Log best model to wandb
                if args.use_wandb:
                    wandb.log({"best_val_accuracy": best_val_accuracy, "best_epoch": epoch + 1})
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
    
    # Synchronize processes before finishing
    if args.distributed:
        dist.barrier()
    
    if is_main_process:
        print("Training finished.")
        if best_epoch != -1:
            print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch+1}")

            # ---- Test evaluation on held-out test subjects with their held-out RUS ----
            best_model_path = os.path.join(args.output_dir, f'best_model_{run_tag}.pth')
            if test_size > 0 and os.path.exists(best_model_path):
                print(f"\nEvaluating on test subjects {test_subjects} using the best checkpoint...")
                torch.serialization.add_safe_globals([argparse.Namespace])
                checkpoint = torch.load(best_model_path, map_location=device)

                test_model = MultimodalTRUSMoEModel(
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
                    output_attention=False,
                ).to(device)
                test_model.load_state_dict(checkpoint['model_state_dict'])

                test_loss, test_acc = evaluate_multimodal(
                    test_model, test_loader, task_criterion, device, args,
                    split_name="Test", current_epoch=None, wandb_prefix="test",
                )
                print(f"Test Accuracy on subjects {test_subjects}: {test_acc:.2f}%")
                if args.use_wandb:
                    wandb.log({"test/final_accuracy": test_acc, "test/final_loss": test_loss})

            # Generate expert activation plots for the best model
            if args.plot_expert_activations and val_size > 0:
                print("\nGenerating expert activation plots for the best multimodal TRUS-MoE model...")

                # Load the best model
                if os.path.exists(best_model_path):
                    # Add argparse.Namespace to safe globals for PyTorch 2.6+ compatibility
                    torch.serialization.add_safe_globals([argparse.Namespace])
                    checkpoint = torch.load(best_model_path, map_location=device)
                    
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
                    
                    # Get a batch of validation data
                    val_batch_modalities = []
                    val_batch_rus = []
                    num_plot_samples = min(args.plot_num_samples, len(val_dataset))
                    
                    for i in range(num_plot_samples):
                        modality_data, rus_data, _ = val_dataset[i]
                        val_batch_modalities.append(modality_data)
                        val_batch_rus.append(rus_data)
                    
                    # Stack into batch format
                    batch_modalities = [[] for _ in range(len(val_batch_modalities[0]))]
                    for sample_modalities in val_batch_modalities:
                        for mod_idx, mod_data in enumerate(sample_modalities):
                            batch_modalities[mod_idx].append(mod_data)
                    
                    # Stack each modality
                    batch_modalities = [torch.stack(mod_list).to(device) for mod_list in batch_modalities]
                    
                    # Stack RUS data
                    batch_rus = {'U': [], 'R': [], 'S': []}
                    for rus_data in val_batch_rus:
                        batch_rus['U'].append(rus_data['U'])
                        batch_rus['R'].append(rus_data['R'])
                        batch_rus['S'].append(rus_data['S'])
                    batch_rus = {k: torch.stack(v).to(device) for k, v in batch_rus.items()}
                    
                    # Generate plots
                    plot_save_dir = os.path.join(args.output_dir, 'expert_activation_plots')
                    
                    try:
                        analyze_expert_activations(
                            trus_model=plot_model,
                            baseline_model=None,
                            data_batch=batch_modalities,
                            rus_values=batch_rus,
                            modality_names=modality_names,
                            save_dir=plot_save_dir,
                            seed=args.seed,
                            subject=format_subject_tag(val_subjects)
                        )
                        print(f"Expert activation plots saved to {plot_save_dir}")
                    except Exception as e:
                        print(f"Error generating expert activation plots: {e}")
                        
                else:
                    print(f"Best model checkpoint not found at {best_model_path}")
            
        else:
            print("Training finished (no validation performed or no improvement).")
    
    # Finish wandb run
    if args.use_wandb and is_main_process:
        wandb.finish()
    
    # Clean up distributed training
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multimodal TRUS-MoE Model on PAMAP2 Data')
    
    # Data args
    parser.add_argument('--train_subjects', type=str, default="1-6",
                        help='Subjects used for training (e.g. "1-6" or "1,2,3,4,5,6")')
    parser.add_argument('--val_subjects', type=str, default="7",
                        help='Subjects used for validation (e.g. "7")')
    parser.add_argument('--test_subjects', type=str, default="8-9",
                        help='Subjects held out for testing (e.g. "8-9" or "8,9")')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATASET_DIR,
                       help='Directory containing PAMAP2 .dat files')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length (T)')
    parser.add_argument('--window_step', type=int, default=50, help='Step size for sliding window')
    
    # RUS Data Args
    parser.add_argument('--rus_file_pattern', type=str, default=DEFAULT_RUS_FILE_PATTERN, 
                       help='Pattern for locating the .npy RUS file')
    parser.add_argument('--rus_max_lag', type=int, default=10, help='Max lag used in RUS file')
    parser.add_argument('--rus_method', type=str, default='batch', help='Method used in RUS file')
    parser.add_argument('--rus_discrim_epochs', type=int, default=30, help='Discriminator epochs used in RUS file')
    parser.add_argument('--rus_ce_epochs', type=int, default=20, help='CE alignment epochs used in RUS file')
    parser.add_argument('--rus_n_batches', type=int, default=1, help='Number of batches used in RUS file')
    parser.add_argument('--rus_batch_size', type=int, default=512, help='Batch size used in RUS file')
    parser.add_argument('--rus_seed', type=int, default=42, help='Seed used in RUS file')

    # Model architecture args
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Feed-forward dimension')
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
    parser.add_argument('--moe_router_gru_hidden_dim', type=int, default=64, help='GRU hidden dim in router')
    parser.add_argument('--moe_router_token_processed_dim', type=int, default=64, 
                       help='Token processing dim in router')
    parser.add_argument('--moe_router_attn_key_dim', type=int, default=32, help='Attention key dim in router')
    parser.add_argument('--moe_router_attn_value_dim', type=int, default=32, 
                       help='Attention value dim in router')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, 
                       help='Max norm for gradient clipping (0 to disable)')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use cosine annealing LR scheduler')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', 
                       help='Use gradient checkpointing to save memory')
    
    # Loss args
    parser.add_argument('--threshold_u', type=float, default=0.5, help='Threshold for uniqueness loss')
    parser.add_argument('--threshold_r', type=float, default=0.1, help='Threshold for redundancy loss')
    parser.add_argument('--threshold_s', type=float, default=0.1, help='Threshold for synergy loss')
    parser.add_argument('--lambda_u', type=float, default=1, help='Weight for uniqueness loss')
    parser.add_argument('--lambda_r', type=float, default=1, help='Weight for redundancy loss')
    parser.add_argument('--lambda_s', type=float, default=1, help='Weight for synergy loss')
    parser.add_argument('--lambda_load', type=float, default=0.02, help='Weight for load balancing loss')
    parser.add_argument('--epsilon_loss', type=float, default=1e-8, help='Epsilon for loss stability')
    
    # System args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--cuda_device', type=int, default=0, help='Specific GPU to use')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                       help='Directory to save results/models')
    
    # Distributed training args
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    
    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='pamap-multimodal-trus-moe', 
                       help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb')
    
    # Expert activation plotting args
    parser.add_argument('--plot_expert_activations', action='store_true', help='Generate expert activation plots after training')
    parser.add_argument('--plot_num_samples', type=int, default=32, help='Number of samples to use for expert activation plotting')
    
    args = parser.parse_args()
    main(args)
