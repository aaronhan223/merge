import pandas as pd
import numpy as np
import math
import os
import sys
import argparse
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pid.temporal_pid_multivariate import multi_lag_analysis
import matplotlib.pyplot as plt
import itertools
import pdb

def parse_subject_list(s):
    """Parses a subject specifier like '1', '1,2,3', or '1-9' into a list of ints."""
    s = str(s).strip()
    if '-' in s and ',' not in s:
        start, end = s.split('-')
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(',') if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description='PAMAP2 dataset multimodal RUS analysis with PID')
    parser.add_argument('--dataset_dir', type=str, default="/cis/home/xhan56/pamap/PAMAP2_Dataset/Protocol",
                        help='Directory containing PAMAP2 dataset files')
    parser.add_argument('--output_dir', type=str, default="./results/pamap",
                        help='Directory to save analysis results')
    parser.add_argument('--subject_ids', type=str, default="1-9",
                        help='Subject IDs to analyze. Accepts a single id ("1"), a comma-'
                             'separated list ("1,2,3"), or a range ("1-9"). RUS values are '
                             'computed independently for each subject.')
    parser.add_argument('--max_lag', type=int, default=10,
                        help='Maximum lag for temporal PID analysis')
    parser.add_argument('--bins', type=int, default=4,
                        help='Number of bins for discretization (reduced for multivariate)')
    parser.add_argument('--dominance_threshold', type=float, default=0.4,
                        help='Threshold for a PID term to be considered dominant')
    parser.add_argument('--dominance_percentage', type=float, default=0.9,
                        help='Percentage of lags a term must dominate to be considered dominant overall')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'joint', 'cvxpy', 'batch',
                                 'multiscale_batch'],
                        help='PID estimation method. "batch" recomputes per '
                             'time lag; "multiscale_batch" trains a single '
                             'lag-conditioned model across all lags '
                             '(Algorithms 1-3 of the MERGE paper). '
                             '"auto" chooses based on dimensionality.')
    
    # General parameters
    parser.add_argument('--batch_size', type=int, default=512,
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
    parser.add_argument('--lag_embed_dim', type=int, default=8,
                        help='Lag embedding dimension for multiscale_batch method')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help='Steps per epoch for multiscale_batch (defaults '
                             'to smallest_lag_dataset // batch_size)')
    parser.add_argument('--discrim_epochs', type=int, default=20,
                        help='Number of epochs for discriminator training in batch method')
    parser.add_argument('--ce_epochs', type=int, default=10,
                        help='Number of epochs for CE alignment training in batch method')
    
    # CVXPY method specific parameters
    parser.add_argument('--regularization', type=float, default=1e-6,
                        help='Regularization parameter for CVXPY method')
    
    return parser.parse_args()

def get_pamap_column_names():
    """Returns the standard column names for PAMAP2 dataset files."""
    columns = ['timestamp', 'activity_id', 'heart_rate']
    imu_locs = ['hand', 'chest', 'ankle']
    imu_sensors = ['temp', 'acc16g_x', 'acc16g_y', 'acc16g_z',
                   'acc6g_x', 'acc6g_y', 'acc6g_z',
                   'gyro_x', 'gyro_y', 'gyro_z',
                   'mag_x', 'mag_y', 'mag_z',
                   'orient_w', 'orient_x', 'orient_y', 'orient_z']

    for loc in imu_locs:
        for sensor in imu_sensors:
            col_name = f"{sensor}_{loc}"
            columns.append(col_name)
    return columns

def load_pamap_data(subject_id, data_dir):
    """Loads data for a specific subject from the PAMAP2 dataset."""
    file_path = os.path.join(data_dir, f"subject10{subject_id}.dat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found for subject {subject_id} at {file_path}")

    print(f"Loading data for subject {subject_id} from {file_path}...")
    df = pd.read_csv(file_path, sep='\s+', header=None, names=get_pamap_column_names())
    print(f"Loaded data shape: {df.shape}")
    return df

def preprocess_pamap_data(df):
    """Preprocesses the loaded PAMAP2 data, keeping all sensor columns."""
    print("Preprocessing data...")
    essential_cols = ['timestamp', 'activity_id']
    sensor_cols = [col for col in df.columns if col not in essential_cols and 'orient' not in col]
    if 'heart_rate' not in df.columns:
        sensor_cols.insert(0, 'heart_rate')
    relevant_cols = essential_cols + sensor_cols
    df_processed = df[relevant_cols].copy()
    print(f"NaN counts before interpolation: {df_processed.isnull().sum()[df_processed.isnull().sum() > 0]}")
    df_processed = df_processed.interpolate(method='linear', limit_direction='both')

    if df_processed.isnull().sum().sum() > 0:
        print("Warning: NaNs still present after interpolation. Dropping rows with NaNs.")
        df_processed.dropna(inplace=True)

    df_processed = df_processed[df_processed['activity_id'] != 0]

    df_processed['activity_id'] = df_processed['activity_id'].astype(int)
    for col in sensor_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(float)

    print(f"Preprocessing complete. Data shape: {df_processed.shape}")
    print(f"Unique activities remaining: {df_processed['activity_id'].unique()}")
    print(f"Available sensor columns for analysis: {sensor_cols}")
    return df_processed, sensor_cols

def categorize_pamap_sensors(sensor_columns):
    """
    Categorizes PAMAP sensor columns by body location/modality.
    
    Expected sensor naming convention:
    - {sensor_type}_{location}
    - e.g., acc16g_x_chest, gyro_y_hand, mag_z_ankle
    
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
        if 'heart' in col_lower or col_lower == 'heart_rate':
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
            print(f"Warning: Could not categorize sensor column: {col}")
    
    # Remove empty modalities
    modality_sensors = {k: v for k, v in modality_sensors.items() if v}
    
    return modality_sensors

def analyze_subject(subject_id, args, device):
    """Run multimodal RUS analysis for a single subject and persist the results."""
    print(f"\n========== Subject {subject_id} ==========")

    try:
        df = load_pamap_data(subject_id, args.dataset_dir)
    except FileNotFoundError as e:
        print(e)
        return False

    df_processed, sensor_columns = preprocess_pamap_data(df)

    # Check if df_processed is empty
    if isinstance(df_processed, pd.DataFrame) and df_processed.empty:
        print(f"No data remaining after preprocessing for subject {subject_id}. Skipping.")
        return False
    elif isinstance(df_processed, np.ndarray) and df_processed.size == 0:
        print(f"No data remaining after preprocessing for subject {subject_id}. Skipping.")
        return False

    # Determine unique activities and create mapping to 0-based indices
    if isinstance(df_processed, pd.DataFrame):
        unique_activities = sorted([act for act in df_processed['activity_id'].unique() if act != 0])
        activity_map = {activity_id: i for i, activity_id in enumerate(unique_activities)}
        num_classes = len(activity_map)
        print(f"\nFound {num_classes} activities: {unique_activities}")
        print(f"Activity mapping: {activity_map}")

        # Remap activity IDs to a new column `activity_label`
        df_processed['activity_label'] = df_processed['activity_id'].map(activity_map)
        df_processed.dropna(subset=['activity_label'], inplace=True)
        df_processed['activity_label'] = df_processed['activity_label'].astype(int)
    else:
        # Handle numpy array case for activity mapping if necessary
        print("Warning: Activity remapping for numpy array not fully implemented.")
        unique_activities = sorted([act for act in np.unique(df_processed[:, 1]) if act != 0])
        activity_map = {activity_id: i for i, activity_id in enumerate(unique_activities)}
        print(f"Activity mapping for array: {activity_map}")

    # Categorize sensors into modalities
    modality_sensors = categorize_pamap_sensors(sensor_columns)
    
    if not modality_sensors:
        print(f"Error: No sensors could be categorized into modalities for subject {subject_id}. Skipping.")
        return False
    
    print(f"\nFound {len(modality_sensors)} modalities:")
    for mod_name, sensors in modality_sensors.items():
        print(f"  {mod_name}: {len(sensors)} sensors - {sensors[:3]}{'...' if len(sensors) > 3 else ''}")

    # Access activity_id correctly based on whether df_processed is DataFrame or array
    if isinstance(df_processed, pd.DataFrame):
        Y = df_processed['activity_label'].values # Use remapped labels
    else:
        # Find activity_id column index if it's an array and map it
        activity_col = df_processed[:, 1]  # Assuming activity_id is the second column
        # Vectorized mapping using the created map
        map_func = np.vectorize(activity_map.get)
        Y = map_func(activity_col)
        # Ensure Y is integer type and handle potential Nones if a key was not in map
        valid_y_mask = Y != None
        Y = Y[valid_y_mask].astype(int)
        # This might cause a mismatch in length, so we need to filter X data as well
        df_processed = df_processed[valid_y_mask]

    if len(Y) <= args.max_lag:
        print(f"Error: Time series length ({len(Y)}) is not sufficient for max_lag ({args.max_lag}) for subject {subject_id}. Skipping.")
        return False

    # Generate pairs of modalities
    modality_names = list(modality_sensors.keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))
    print(f"\nGenerated {len(modality_pairs)} pairs of modalities for analysis: {modality_pairs}")

    # Prepare data for each modality (multivariate arrays)
    modality_data = {}
    for mod_name, sensors in modality_sensors.items():
        # Get only the sensors that exist in the dataframe
        if isinstance(df_processed, pd.DataFrame):
            existing_sensors = [s for s in sensors if s in df_processed.columns]
            if existing_sensors:
                modality_data[mod_name] = df_processed[existing_sensors].values
                print(f"Modality '{mod_name}': {modality_data[mod_name].shape} (samples, features)")
            else:
                print(f"Warning: No existing sensors found for modality {mod_name}")
                continue
        else:
            # Handle array case - would need column mapping
            print(f"Warning: Cannot extract modality data from array format")
            continue

    dominant_pid_results = []  # List to store results where a term is dominant
    all_pid_results = []  # List to store results for all modality pairs

    for i, (mod1, mod2) in enumerate(modality_pairs):
        print(f"\n--- Analyzing Modality Pair {i+1}/{len(modality_pairs)}: {mod1} vs {mod2} ---")

        if mod1 not in modality_data or mod2 not in modality_data:
            print(f"Warning: Missing data for pair ({mod1}, {mod2}). Skipping.")
            continue

        X1 = modality_data[mod1]  # Shape: (n_samples, n_features_mod1)
        X2 = modality_data[mod2]  # Shape: (n_samples, n_features_mod2)
        
        if len(X1) != len(Y) or len(X2) != len(Y):
            print(f"Warning: Length mismatch for pair ({mod1}, {mod2}). Skipping.")
            print(f"Len X1: {len(X1)}, Len X2: {len(X2)}, Len Y: {len(Y)}")
            continue

        print(f"Starting Multimodal Temporal PID analysis for Subject {subject_id}...")
        print(f"X1 ({mod1}): {X1.shape} - {X1.shape[1]} features")
        print(f"X2 ({mod2}): {X2.shape} - {X2.shape[1]} features")
        print(f"Y: activity_id ({len(Y)} samples)")
        print(f"Max Lag: {args.max_lag}, Bins: {args.bins}")

        # try:
        # Use multi_lag_analysis which will automatically detect multivariate input
        # and use temporal_pid_multivariate internally
        # Select method based on user choice or data dimensionality
        n_total_features = X1.shape[1] + X2.shape[1]
        
        if args.method == 'auto':
            # Automatically select based on dimensionality
            if n_total_features <= 10:
                method = 'joint'
                print(f"Auto-selected 'joint' method (total features: {n_total_features})")
            elif n_total_features <= 20:
                method = 'cvxpy'
                print(f"Auto-selected 'cvxpy' method (total features: {n_total_features})")
            else:
                method = 'batch'
                print(f"Auto-selected 'batch' method (total features: {n_total_features})")
        else:
            method = args.method
            print(f"Using user-specified '{method}' method (total features: {n_total_features})")

        # Call multi_lag_analysis with appropriate parameters
        if method == 'joint':
            pid_results = multi_lag_analysis(X1, X2, Y, max_lag=args.max_lag, bins=args.bins, method=method)
        elif method == 'cvxpy':
            pid_results = multi_lag_analysis(X1, X2, Y, max_lag=args.max_lag, bins=args.bins,
                                            method=method, regularization=args.regularization)
        elif method == 'batch':
            pid_results = multi_lag_analysis(X1, X2, Y, max_lag=args.max_lag, bins=args.bins,
                                            method=method,
                                            batch_size=min(args.batch_size, len(Y)//2),
                                            n_batches=args.n_batches,
                                            seed=args.seed,
                                            device=device,
                                            hidden_dim=args.hidden_dim,
                                            layers=args.layers,
                                            activation=args.activation,
                                            lr=args.lr,
                                            embed_dim=args.embed_dim,
                                            discrim_epochs=args.discrim_epochs,
                                            ce_epochs=args.ce_epochs)
        elif method == 'multiscale_batch':
            pid_results = multi_lag_analysis(X1, X2, Y, max_lag=args.max_lag, bins=args.bins,
                                            method=method,
                                            batch_size=min(args.batch_size, len(Y)//2),
                                            seed=args.seed,
                                            device=device,
                                            hidden_dim=args.hidden_dim,
                                            layers=args.layers,
                                            activation=args.activation,
                                            lr=args.lr,
                                            embed_dim=args.embed_dim,
                                            lag_embed_dim=args.lag_embed_dim,
                                            discrim_epochs=args.discrim_epochs,
                                            ce_epochs=args.ce_epochs,
                                            steps_per_epoch=args.steps_per_epoch)
        # except Exception as e:
        #     print(f"Error during PID analysis for pair ({mod1}, {mod2}): {e}")
        #     continue

        # --- Analyze dominance across all lags as a unit ---
        lags = pid_results.get('lag', range(args.max_lag + 1))
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
                'feature_pair': (mod1, mod2),  # Now modality names instead of sensor names
                'avg_metrics': avg_metrics,
                'lag_results': lag_results,
                'modality1_features': modality_sensors[mod1],
                'modality2_features': modality_sensors[mod2],
                'n_features_mod1': len(modality_sensors[mod1]),
                'n_features_mod2': len(modality_sensors[mod2])
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
                        'modality1_features': modality_sensors[mod1],
                        'modality2_features': modality_sensors[mod2],
                        'n_features_mod1': len(modality_sensors[mod1]),
                        'n_features_mod2': len(modality_sensors[mod2])
                    })
                    break  # We've found the dominant term, no need to check others

    print(f"\nAnalysis complete for all {len(modality_pairs)} modality pairs.")

    # --- Save dominant PID results ---
    if dominant_pid_results:
        output_filename = f'pamap_subject{subject_id}_dominant_lag{args.max_lag}_{args.method}_bins{args.bins}_thresh{args.dominance_threshold:.1f}_pct{int(args.dominance_percentage*100)}_discrim_epochs{args.discrim_epochs}_ce_epochs{args.ce_epochs}_n_batches{args.n_batches}_batch_size{args.batch_size}_seed{args.seed}.npy'
        output_path = os.path.join(args.output_dir, output_filename)
        print(f"Saving {len(dominant_pid_results)} dominant multimodal PID results to {output_path}...")
        np.save(output_path, dominant_pid_results, allow_pickle=True)
        print("Saving dominant modality pairs complete.")

    # --- Save all PID results ---
    if all_pid_results:
        all_output_filename = f'pamap_subject{subject_id}_lag{args.max_lag}_{args.method}_discrim_epochs{args.discrim_epochs}_ce_epochs{args.ce_epochs}_n_batches{args.n_batches}_batch_size{args.batch_size}_seed{args.seed}.npy'
        all_output_path = os.path.join(args.output_dir, all_output_filename)
        print(f"Saving {len(all_pid_results)} PID results for all modality pairs to {all_output_path}...")
        np.save(all_output_path, all_pid_results, allow_pickle=True)
        print("Saving all modality pairs complete.")

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

    print("\n--- Modality Information Summary ---")
    for mod_name, sensors in modality_sensors.items():
        print(f"Modality '{mod_name}': {len(sensors)} features")
        if len(sensors) <= 5:
            print(f"  Features: {sensors}")
        else:
            print(f"  Sample features: {sensors[:5]}...")
    print("------------------------------------")

    return True


def main():
    """Main function: iterate over subjects and run RUS analysis per subject."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up GPU device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    # Set random seed for reproducibility (done once; per-subject reseed below for parity)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    subject_ids = parse_subject_list(args.subject_ids)
    print(f"Computing RUS values for subjects: {subject_ids}")

    succeeded, skipped = [], []
    for sid in subject_ids:
        # Reseed per subject so each run is reproducible independent of order
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if analyze_subject(sid, args, device):
            succeeded.append(sid)
        else:
            skipped.append(sid)

    print(f"\n========== RUS Computation Summary ==========")
    print(f"Subjects processed successfully: {succeeded}")
    if skipped:
        print(f"Subjects skipped due to errors: {skipped}")
    print("=============================================")


if __name__ == "__main__":
    main()
