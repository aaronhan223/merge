"""
This script computes the RUS of the time series of a stay in MIMIC-IV
Example usage for in-hospital mortality:
1. Linear interpolation + timestep pooling (no pooling)
python mimiciv_rus_multimodal.py --train_dataset_path /path/to/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl --task ihm --linear_interpolation --seq_len 48 --num_lags 6

2. Mean pooling
python mimiciv_rus_multimodal.py --train_dataset_path /path/to/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl --task ihm --sequence_pooling mean --seq_len 48 --num_lags 6

Example usage for length of stay:
python mimiciv_rus_multimodal.py --train_dataset_path /path/to/train_los-cxr-notes-missingInd-standardized_stays.pkl --task los --linear_interpolation --seq_len 48 --num_lags 6
python mimiciv_rus_multimodal.py --train_dataset_path /path/to/train_los-cxr-notes-missingInd-standardized_stays.pkl --task los --sequence_pooling mean --seq_len 48 --num_lags 6
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple
import itertools
import torch
from tqdm import tqdm
from pid.temporal_pid_multi_sequence import temporal_pid_label_multi_sequence_multi_lag

def align_multimodal_irg_ts(reg_ts: np.ndarray, multimodal_irg_times_feats: Dict[str, List[Tuple[float, np.ndarray]]], modality_dim_dict: Dict[str, int], interval_length=1, linear_interpolation=False):
    """
    Align the regular time series (labs + vitals) with multimodal irrregular time series features (notes + cxr).
    Args:
        reg_ts: The regular time series, shape: (T, D). T is the number of time steps, D is the number of features.
        multimodal_irg_times_feats: A dictionary of multimodal irregular time series features.
            The key is the modality name, the value is a list of tuples, each containing the timestamp and the feature vector.
        modality_dim_dict: A dictionary of modality names and their dimensions.
        interval_length: The length of the intervals between the regular time series time steps for us to plug in the multimodal features.
        linear_interpolation: If True, fill missing timesteps (where mask=0) with linear interpolation and set mask to all ones.
    Returns:
        A dictionary of aligned time series and masks. The key is the modality name, the value is a tuple of:
        - aligned time series of shape: (T, number of features in the modality)
        - binary mask of shape: (T,) indicating which timesteps have actual irregular data (1) vs zero-filled (0).
          If linear_interpolation=True, mask will be all ones after interpolation.
    """
    num_time_steps = len(reg_ts)
    multimodal_reg_ts = {}
    
    for modality, feats_list in multimodal_irg_times_feats.items():
        # Get number of features for this modality
        num_features = modality_dim_dict[modality]
        aligned_ts = np.zeros((num_time_steps, num_features))
        mask = np.zeros(num_time_steps, dtype=bool)  # Binary mask for valid timesteps
        
        # Handle empty feature lists
        if not feats_list:
            print(f"Warning: No features found for modality {modality}, using all zeros")
            # Keep aligned_ts as all zeros and mask as all False
            multimodal_reg_ts[modality] = (aligned_ts, mask)
            continue
        
        # First pass: collect all features by index
        features_by_index = {}
        for time, feats in feats_list:
            index = int(time / interval_length)
            
            # Skip if index is out of bounds
            if index >= num_time_steps or index < 0:
                print(f"Warning: Time {time} maps to index {index} which is out of bounds {num_time_steps} for modality {modality}")
                continue
                
            if index not in features_by_index:
                features_by_index[index] = []
            features_by_index[index].append(feats)
        
        # Second pass: average features at each index and set mask
        for index, feat_list in features_by_index.items():
            # if len(feat_list) > 1:
            #     print(f"Averaging {len(feat_list)} measurements at index {index} for modality {modality}")
            aligned_ts[index, :] = np.mean(feat_list, axis=0)
            mask[index] = True  # Mark this timestep as having actual data
        
        # Apply linear interpolation if requested (equivalent to pandas interpolate method='linear', limit_direction='both')
        if linear_interpolation:
            # Only interpolate if we have at least 1 data point
            valid_indices = np.where(mask)[0]
            if len(valid_indices) >= 1:
                # For each feature dimension, interpolate missing values
                for feat_idx in range(num_features):
                    # Perform linear interpolation equivalent to pandas interpolate(method='linear', limit_direction='both')
                    # np.interp handles both multi-point interpolation and single-point constant fill
                    valid_times = valid_indices.astype(float)
                    valid_values = aligned_ts[valid_indices, feat_idx]
                    
                    # Interpolate for all timesteps (including extrapolation at both ends)
                    all_times = np.arange(num_time_steps, dtype=float)
                    interpolated_values = np.interp(all_times, valid_times, valid_values)
                    
                    # Update the aligned time series
                    aligned_ts[:, feat_idx] = interpolated_values
                
                # Set mask to all ones since all timesteps now have interpolated data
                mask = np.ones(num_time_steps, dtype=bool)
            # If no valid data points, keep original aligned_ts (all zeros) and mask (all False)
            
        multimodal_reg_ts[modality] = (aligned_ts, mask)
        
    return multimodal_reg_ts

def preprocess_mimiciv_data(stays: List[Dict], modality_dim_dict: Dict[str, int], num_subsample_stays: int = None, linear_interpolation: bool = False) -> List[Dict]:
    """
    Preprocess the MIMIC-IV data.
    Args:
        stays: List of stays.
        modality_dim_dict: Dictionary of modality names and their dimensions.
        num_subsample_stays: Number of stays to randomly sample for analysis. If None, use all eligible stays.
        linear_interpolation: If True, apply linear interpolation to fill missing timesteps in multimodal irregular time series.
    Returns:
        all_multimodal_reg_ts: List of multimodal regular time series.
        all_labels: List of labels.
    """
    eligible_stays = []
    for stay in stays:
        if len(stay['ts_tt']) > 12:
            eligible_stays.append(stay)
    
    print(f"Found {len(eligible_stays)} eligible stays (more than 12 labs/vitals record times)")
    
    # Randomly sample num_subsample_stays from eligible stays
    if num_subsample_stays is None:
        print("Using all eligible stays for analysis.")
        selected_stays = eligible_stays
    elif num_subsample_stays > len(eligible_stays):
        print(f"Warning: Requested {num_subsample_stays} stays but only {len(eligible_stays)} are eligible. Using all eligible stays.")
        selected_stays = eligible_stays
    else:
        selected_stays = np.random.choice(eligible_stays, size=num_subsample_stays, replace=False).tolist()
    
    print(f"Selected {len(selected_stays)} stays for analysis")

    # Process all selected stays

    all_multimodal_reg_ts = []
    all_labels = []
    
    for i, stay in enumerate(selected_stays):
        
        multimodal_irg_ts = {
            'notes': [(stay['text_time'][j], stay['text_embeddings'][j]) for j in range(len(stay['text_time']))],
            'cxr': [(stay['cxr_time'][j], stay['cxr_feats'][j]) for j in range(len(stay['cxr_time']))]
        }
        
        multimodal_reg_ts = align_multimodal_irg_ts(stay['reg_ts'], multimodal_irg_ts, modality_dim_dict, linear_interpolation=linear_interpolation)
        # Add labs_vitals with a full mask (all timesteps are valid for regular time series)
        multimodal_reg_ts['labs_vitals'] = (stay['reg_ts'], np.ones(len(stay['reg_ts']), dtype=bool))
        
        all_multimodal_reg_ts.append(multimodal_reg_ts)
        all_labels.append(stay['label'])
    
    return all_multimodal_reg_ts, all_labels

def main(args):
    # Set up device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    # Randomly sample num_subsample_stays from eligible stays
    np.random.seed(args.seed)

    train_stays = pickle.load(open(args.train_dataset_path, 'rb'))
    
    modality_dim_dict = {'labs_vitals': 30,
                         'cxr': 1024,
                         'notes': 768}
    all_multimodal_reg_ts, all_labels = preprocess_mimiciv_data(train_stays, modality_dim_dict, args.num_subsample_stays, args.linear_interpolation)    
    
    # Get modality names from first stay (assuming all stays have same modalities)
    modality_names = list(modality_dim_dict.keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))
    print(f"\nGenerated {len(modality_pairs)} pairs of modalities for analysis: {modality_pairs}")
    
    dominant_pid_results = []
    all_pid_results = []
    
    # Prepare data for temporal_pid_label_multi_sequence_batch
    for mod1, mod2 in modality_pairs:
        print(f"\n--- Analyzing Modality Pair {mod1} vs {mod2} ---")
        
        # Extract time series for this modality pair across all stays
        # Each modality now returns (time_series, mask) tuple, so extract just the time series
        X1_list = [stay_data[mod1][0] for stay_data in all_multimodal_reg_ts]  # Extract time series
        X2_list = [stay_data[mod2][0] for stay_data in all_multimodal_reg_ts]  # Extract time series
        Y_list = all_labels
        
        X1_masks = [stay_data[mod1][1] for stay_data in all_multimodal_reg_ts]  # Extract masks
        X2_masks = [stay_data[mod2][1] for stay_data in all_multimodal_reg_ts]  # Extract masks
        
        print(f"Number of stays: {len(X1_list)}")
        print(f"X1 ({mod1}) shapes: {[x.shape for x in X1_list[:3]]}{'...' if len(X1_list) > 3 else ''}")
        print(f"X2 ({mod2}) shapes: {[x.shape for x in X2_list[:3]]}{'...' if len(X2_list) > 3 else ''}")
        print(f"Labels: {Y_list[:10]}{'...' if len(Y_list) > 10 else ''}")
        
        # Use multi-lag analysis
        pid_results = temporal_pid_label_multi_sequence_multi_lag(
            X1_list, X2_list, Y_list, X1_masks, X2_masks,
            seq_len=args.seq_len,
            num_lags=args.num_lags,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            discrim_epochs=args.discrim_epochs,
            ce_epochs=args.ce_epochs,
            seed=args.seed,
            device=device,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            activation=args.activation,
            lr=args.lr,
            embed_dim=args.embed_dim,
            n_labels=len(np.unique(Y_list)),
            sequence_pooling=args.sequence_pooling
        )
        print(f"PID Results for {mod1} vs {mod2}: {len(pid_results['lag'])} lags analyzed")

        # --- Analyze dominance across all lags as a unit ---
        lags = pid_results.get('lag', [])
        dominant_counts = {'R': 0, 'U1': 0, 'U2': 0, 'S': 0}
        total_valid_lags = 0
        
        lag_results = []  # Store results for all lags for this pair
        
        for lag_idx, lag in enumerate(lags):
            try:
                r = pid_results['redundancy'][lag_idx]
                u1 = pid_results['unique_x1'][lag_idx]
                u2 = pid_results['unique_x2'][lag_idx]
                s = pid_results['synergy'][lag_idx]
                mi = pid_results['total_di'][lag_idx]

                if mi > 1e-9:  # Avoid division by zero or near-zero MI
                    total_valid_lags += 1
                    
                    # Get normalized values for each term
                    r_norm = r / mi
                    u1_norm = u1 / mi
                    u2_norm = u2 / mi
                    s_norm = s / mi
                    
                    # Find the term with the highest value
                    norm_values = {
                        'R': r_norm,
                        'U1': u1_norm,
                        'U2': u2_norm,
                        'S': s_norm
                    }
                    # Get key with maximum value
                    max_term = None
                    max_value = -1
                    for term, value in norm_values.items():
                        if value > max_value:
                            max_value = value
                            max_term = term
                    
                    # If highest and above threshold, count it
                    if max_term and max_value > args.dominance_threshold:
                        dominant_counts[max_term] += 1
                    
                    # Store this lag's result
                    lag_results.append({
                        'lag': lag,
                        'R_value': r,
                        'U1_value': u1,
                        'U2_value': u2,
                        'S_value': s,
                        'MI_value': mi,
                        'R_norm': r_norm,
                        'U1_norm': u1_norm,
                        'U2_norm': u2_norm,
                        'S_norm': s_norm
                    })
            except IndexError:
                print(f"Warning: Index out of bounds for lag {lag} (index {lag_idx}) for pair ({mod1}, {mod2}). Skipping lag.")
                continue
            except KeyError as e:
                print(f"Warning: Missing key {e} in pid_results for pair ({mod1}, {mod2}). Skipping dominance check.")
                break  # Stop checking lags for this pair if keys are missing
        
        # Check if we have enough valid lags to evaluate
        if total_valid_lags > 0:
            # Calculate average metrics across all lags
            avg_metrics = {
                'R_value': np.mean([r['R_value'] for r in lag_results]),
                'U1_value': np.mean([r['U1_value'] for r in lag_results]),
                'U2_value': np.mean([r['U2_value'] for r in lag_results]), 
                'S_value': np.mean([r['S_value'] for r in lag_results]),
                'MI_value': np.mean([r['MI_value'] for r in lag_results]),
                'R_norm': np.mean([r['R_norm'] for r in lag_results]),
                'U1_norm': np.mean([r['U1_norm'] for r in lag_results]),
                'U2_norm': np.mean([r['U2_norm'] for r in lag_results]),
                'S_norm': np.mean([r['S_norm'] for r in lag_results])
            }
            
            # Save results for all modality pairs
            all_pid_results.append({
                'feature_pair': (mod1, mod2),
                'avg_metrics': avg_metrics,
                'lag_results': lag_results,
                'n_features_mod1': X1_list[0].shape[1],
                'n_features_mod2': X2_list[0].shape[1]
            })
            
            # Find term that is dominant across at least percentage of the lags
            for term, count in dominant_counts.items():
                dominance_ratio = count / total_valid_lags
                if dominance_ratio >= args.dominance_percentage:
                    print(f"Found dominant term {term} for pair ({mod1}, {mod2}) across {dominance_ratio:.1%} of lags")
                    
                    # Store this pair's result as dominant
                    dominant_pid_results.append({
                        'feature_pair': (mod1, mod2),
                        'dominant_term': term,
                        'dominance_ratio': dominance_ratio,
                        'lags_analyzed': total_valid_lags,
                        'avg_metrics': avg_metrics,
                        'lag_results': lag_results,
                        'n_features_mod1': X1_list[0].shape[1],
                        'n_features_mod2': X2_list[0].shape[1]
                    })
                    break  # We've found the dominant term, no need to check others

    if dominant_pid_results:
        output_filename = f'rus_multimodal_dominant_seq{args.seq_len}_lags{args.num_lags}_thresh{args.dominance_threshold:.1f}_pct{int(args.dominance_percentage*100)}_{args.sequence_pooling}pool.npy'
        output_path = os.path.join(args.output_dir, args.task, output_filename)
        print(f"Saving {len(dominant_pid_results)} dominant multimodal PID results to {output_path}...")
        np.save(output_path, dominant_pid_results, allow_pickle=True)
        print("Saving dominant modality pairs complete.")
    
    if all_pid_results:
        all_output_filename = f'rus_multimodal_all_seq{args.seq_len}_lags{args.num_lags}_{args.sequence_pooling}pool.npy'
        all_output_path = os.path.join(args.output_dir, args.task, all_output_filename)
        print(f"Saving {len(all_pid_results)} PID results for all modality pairs to {all_output_path}...")
        np.save(all_output_path, all_pid_results, allow_pickle=True)
        print("Saving all modality pairs complete.")

    print(f"\nAnalysis complete for all {len(modality_pairs)} modality pairs.")

    # --- Print summary of dominant terms ---
    if dominant_pid_results:
        dominance_counts = {'R': 0, 'U1': 0, 'U2': 0, 'S': 0}
        for result in dominant_pid_results:
            term = result.get('dominant_term')
            if term in dominance_counts:
                dominance_counts[term] += 1

        print("\n--- Multimodal Dominance Summary ---")
        print(f"Total modality pairs with dominant terms: {len(dominant_pid_results)}")
        for term, count in dominance_counts.items():
            print(f"  {term} dominant: {count} pairs")
        print("----------------------------------------")

    else:
        print("No dominant PID terms found with the current threshold and percentage criteria.")

    # --- Print modality information summary ---
    print("\n--- Modality Information Summary ---")
    for mod_name, dim in modality_dim_dict.items():
        print(f"Modality '{mod_name}': {dim} features")
    print("------------------------------------")

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MIMIC-IV RUS computation')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save analysis results')
    parser.add_argument('--task', type=str, choices=['ihm', 'los'], required=True, help='Task to analyze, either in-hospital mortality (ihm) or length of stay (los)')
    parser.add_argument('--train_dataset_path', type=str, required=True,
                        help='Path to the preprocessed MIMIC-IV train dataset')
    parser.add_argument('--num_subsample_stays', type=int, default=None,
                        help='Number of stays to randomly sample for analysis')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Length of sequences for lag computation. If None, inferred from data')
    parser.add_argument('--num_lags', type=int, default=3,
                        help='Number of lags to compute, evenly distributed across the sequence')
    parser.add_argument('--dominance_threshold', type=float, default=0.4,
                        help='Threshold for a PID term to be considered dominant')
    parser.add_argument('--dominance_percentage', type=float, default=0.9,
                        help='Percentage of lags a term must dominate to be considered dominant overall')
    # General parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for batch method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GPU selection
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    
    # BATCH method specific parameters
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of batches for batch method')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension for neural networks in batch method')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers for neural networks in batch method')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh'],
                        help='Activation function for neural networks in batch method')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for neural networks in batch method')
    parser.add_argument('--embed_dim', type=int, default=10,
                        help='Embedding dimension for alignment model in batch method')
    parser.add_argument('--discrim_epochs', type=int, default=100,
                        help='Number of epochs for discriminator training in batch method')
    parser.add_argument('--ce_epochs', type=int, default=10,
                        help='Number of epochs for CE alignment training in batch method')
    parser.add_argument('--sequence_pooling', type=str, default='timestep',
                        choices=['timestep', 'mean'],
                        help='How to process sequences for batch method (timestep, mean)')
    parser.add_argument('--linear_interpolation', action='store_true',
                        help='Apply linear interpolation to fill missing timesteps in irregular time series')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    main(args)
