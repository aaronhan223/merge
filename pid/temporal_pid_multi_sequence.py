import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from estimators.ce_alignment_information import (
    CEAlignmentInformation, Discrim, MultimodalDataset,
    train_discrim, train_ce_alignment, eval_ce_alignment,
)

def temporal_pid_label_multi_sequence_multi_lag(X1_list, X2_list, Y_list, X1_masks, X2_masks, seq_len=None, num_lags=3, batch_size=256, n_batches=10, 
                      discrim_epochs=20, ce_epochs=10, seed=42, device=None,
                      hidden_dim=32, layers=2, activation='relu', lr=1e-3, embed_dim=10, n_labels=None,
                      sequence_pooling='timestep'):
    """
    Compute PID using batch/neural network method for multiple time series sequence/label pairs across multiple lags.
    Parameters:
    -----------
    X1_list, X2_list: List[numpy.ndarray]
        Lists of time series, one per sequence
    Y_list: List[int]
        Classification labels for each sequence
    X1_masks, X2_masks: List[numpy.ndarray]
        Lists of binary masks indicating valid timesteps for each sequence
    seq_len: int
        Length of the sequences. If None, inferred from the data
    num_lags: int
        Number of lags to compute, evenly distributed across the sequence
    batch_size: int
        Batch size for training
    n_batches : int
        Number of batches to average over
    discrim_epochs : int
        Epochs for discriminator training
    ce_epochs : int
        Epochs for alignment training
    seed : int
        Random seed
    device : torch.device
        Device to run computations on
    hidden_dim : int
        Hidden dimension for neural networks
    layers : int
        Number of layers for neural networks
    activation : str
        Activation function for neural networks
    lr : float
        Learning rate for neural networks
    embed_dim : int
        Embedding dimension for alignment model
    n_labels : int
        Number of class labels. If None, inferred from Y_list
    sequence_pooling : str
        How to process sequences: 'timestep' (default, each timestep as sample), 
        'mean' (mean pooling over masked timesteps only)
        
    Returns:
    --------
    results : dict
        PID components for each lag
    """
    # Generate lags based on seq_len and num_lags
    if seq_len is None:
        # Infer seq_len from the data (use minimum length across all sequences)
        seq_len = min(len(x) for x in X1_list)
        print(f"Inferred seq_len from data: {seq_len}")
    
    # Validate that seq_len % num_lags == 0
    if seq_len % num_lags != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by num_lags ({num_lags})")
    
    # Generate evenly spaced lags
    if num_lags == 1:
        lags = [0]
    else:
        lags = [i * seq_len // num_lags for i in range(num_lags)]
    
    print(f"Computing lags: {lags} (seq_len={seq_len}, num_lags={num_lags})")
    
    results = {
        'lag': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_di': [],
        'method': []
    }
    
    for lag in tqdm(lags, desc="Processing lags"):
        try:
            pid_result = temporal_pid_label_multi_sequence_batch(
                X1_list, X2_list, Y_list, X1_masks, X2_masks,
                lag=lag, batch_size=batch_size, n_batches=n_batches,
                discrim_epochs=discrim_epochs, ce_epochs=ce_epochs,
                seed=seed, device=device, hidden_dim=hidden_dim,
                layers=layers, activation=activation, lr=lr,
                embed_dim=embed_dim, n_labels=n_labels,
                sequence_pooling=sequence_pooling
            )
            
            results['lag'].append(lag)
            results['redundancy'].append(pid_result['redundancy'])
            results['unique_x1'].append(pid_result['unique_x1'])
            results['unique_x2'].append(pid_result['unique_x2'])
            results['synergy'].append(pid_result['synergy'])
            results['total_di'].append(pid_result['total_di'])
            results['method'].append(pid_result.get('method', 'batch'))
            
        except Exception as e:
            print(f"Error at lag {lag}: {str(e)}")
            # Append NaN values for failed lags
            results['lag'].append(lag)
            for key in ['redundancy', 'unique_x1', 'unique_x2', 'synergy', 'total_di']:
                results[key].append(np.nan)
            results['method'].append('failed')
    
    return results

def temporal_pid_label_multi_sequence_batch(X1_list, X2_list, Y_list, X1_masks, X2_masks, lag=1, batch_size=256, n_batches=10, 
                      discrim_epochs=20, ce_epochs=10, seed=42, device=None,
                      hidden_dim=32, layers=2, activation='relu', lr=1e-3, embed_dim=10, n_labels=None,
                      sequence_pooling='timestep'):
    """
    Compute PID using batch/neural network method for multiple time series sequence/label pairs.
    Parameters:
    -----------
    X1_list, X2_list: List[numpy.ndarray]
        Lists of time series, one per sequence
    Y_list: List[int]
        Classification labels for each sequence
    X1_masks, X2_masks: List[numpy.ndarray]
        Lists of binary masks indicating valid timesteps for each sequence
    lag: int
        Time lag
    batch_size: int
        Batch size for training
    n_batches : int
        Number of batches to average over
    discrim_epochs : int
        Epochs for discriminator training
    ce_epochs : int
        Epochs for alignment training
    seed : int
        Random seed
    device : torch.device
        Device to run computations on
    hidden_dim : int
        Hidden dimension for neural networks
    layers : int
        Number of layers for neural networks
    activation : str
        Activation function for neural networks
    lr : float
        Learning rate for neural networks
    embed_dim : int
        Embedding dimension for alignment model
    n_labels : int
        Number of class labels. If None, inferred from Y_list
    sequence_pooling : str
        How to process sequences: 'timestep' (default, each timestep as sample), 
        'mean' (mean pooling over masked timesteps only)
        
    Returns:
    --------
    results : dict
        PID components
    """
    # Validate parameters
    if len(X1_masks) != len(X1_list):
        raise ValueError(f"X1_masks length ({len(X1_masks)}) must match X1_list length ({len(X1_list)})")
    if len(X2_masks) != len(X2_list):
        raise ValueError(f"X2_masks length ({len(X2_masks)}) must match X2_list length ({len(X2_list)})")

    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    Y_array = np.array(Y_list)
    unique_labels = np.unique(Y_array)
    actual_n_labels = len(unique_labels)
    if n_labels is None:
        n_labels = actual_n_labels
    else:
        n_labels = max(n_labels, actual_n_labels)
    assert np.all((Y_array >= 0) & (Y_array < n_labels)), f"All labels must be in range [0, {n_labels-1}], but found: {np.unique(Y_array)}" 
    
    # Warn if we have a degenerate case (only one label)
    if actual_n_labels == 1:
        print(f"Warning: Only one unique label found ({unique_labels[0]}). "
              f"This may lead to degenerate distributions. Consider providing multiple classes.")
    
    print(f"Using sequence pooling method: {sequence_pooling}")
    
    # Collect all data points
    all_X1_data = []
    all_X2_data = []
    all_Y_labels = []
    

    for i, (X1, X2, Y) in enumerate(zip(X1_list, X2_list, Y_list)):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)

        if len(X1) != len(X2):
            raise ValueError("X1 and X2 must have the same length")

        if lag == 0:
            X1_past = X1
            X2_past = X2
        else:
            X1_past = X1[:-lag, :]
            X2_past = X2[:-lag, :]
        
        if len(X1_past) == 0:
            continue        

        # Get masks for this sequence
        X1_mask = X1_masks[i]
        X2_mask = X2_masks[i]
        
        # Apply lag to masks as well
        if lag == 0:
            X1_mask_past = X1_mask
            X2_mask_past = X2_mask
        else:
            X1_mask_past = X1_mask[:-lag]
            X2_mask_past = X2_mask[:-lag]

        if sequence_pooling == 'timestep':
            # Treat each time step as independent data point
            all_X1_data.extend(X1_past)
            all_X2_data.extend(X2_past)
            all_Y_labels.extend([Y] * len(X1_past))
        elif sequence_pooling == 'mean':
            # Mean pooling approach: only average over masked (valid) timesteps
            X1_valid_mask = X1_mask_past
            X2_valid_mask = X2_mask_past
            
            if np.any(X1_valid_mask) and np.any(X2_valid_mask):
                X1_pooled = np.mean(X1_past[X1_valid_mask], axis=0)
                X2_pooled = np.mean(X2_past[X2_valid_mask], axis=0)
                all_X1_data.append(X1_pooled)
                all_X2_data.append(X2_pooled)
                all_Y_labels.append(Y)
            # If no valid timesteps in either modality, skip this sequence
        else:
            raise ValueError(f"Unknown sequence_pooling method: {sequence_pooling}. Choose from 'timestep', 'mean'")
    all_X1_data = np.array(all_X1_data)
    all_X2_data = np.array(all_X2_data)
    all_Y_labels = np.array(all_Y_labels)

    X1_tensor = torch.tensor(all_X1_data, dtype=torch.float32, device=device)
    X2_tensor = torch.tensor(all_X2_data, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(all_Y_labels, dtype=torch.long, device=device)

    X1_tensor = X1_tensor.view(-1, X1_tensor.shape[-1])
    X2_tensor = X2_tensor.view(-1, X2_tensor.shape[-1])
    Y_tensor = Y_tensor.view(-1)
    
    print(f"Using sequence pooling method: {sequence_pooling}")


    # Create train/test split
    n_samples = len(Y_tensor)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_ds = MultimodalDataset(
        [X1_tensor[train_idx], X2_tensor[train_idx]], 
        Y_tensor[train_idx]
    )
    test_ds = MultimodalDataset(
        [X1_tensor[test_idx], X2_tensor[test_idx]], 
        Y_tensor[test_idx]
    )
    
    all_results = []
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        # Sample subset of data for this batch
        if len(train_ds) > batch_size * 10:
            batch_indices = np.random.choice(len(train_ds), 
                                           size=min(batch_size * 10, len(train_ds)), 
                                           replace=False)
            batch_X1 = X1_tensor[train_idx[batch_indices]]
            batch_X2 = X2_tensor[train_idx[batch_indices]]
            batch_Y = Y_tensor[train_idx[batch_indices]]
            
            batch_train_ds = MultimodalDataset([batch_X1, batch_X2], batch_Y)
        else:
            batch_train_ds = train_ds
            
        # Train discriminators
        x1_dim = X1_tensor.shape[1]
        x2_dim = X2_tensor.shape[1]
        
        model_discrim_1 = Discrim(x_dim=x1_dim, hidden_dim=hidden_dim, num_labels=n_labels, 
                                 layers=layers, activation=activation).to(device)
        model_discrim_2 = Discrim(x_dim=x2_dim, hidden_dim=hidden_dim, num_labels=n_labels, 
                                 layers=layers, activation=activation).to(device)
        model_discrim_12 = Discrim(x_dim=x1_dim + x2_dim, hidden_dim=hidden_dim, 
                                  num_labels=n_labels, layers=layers, activation=activation).to(device)
        
        # Train each discriminator
        for model, data_type in [
            (model_discrim_1, ([1], [0])),
            (model_discrim_2, ([2], [0])),
            (model_discrim_12, ([1], [2], [0])),
        ]:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_loader = DataLoader(batch_train_ds, shuffle=True, 
                                    batch_size=min(batch_size, len(batch_train_ds)),
                                    num_workers=0)
            train_discrim(model, train_loader, optimizer, 
                         data_type=data_type, num_epoch=discrim_epochs, device=device)
            model.eval()
    
        # Compute prior P(Y)
        p_y = torch.zeros(n_labels)
        for i in range(n_labels):
            p_y[i] = (Y_tensor[train_idx] == i).float().mean()
        p_y = p_y.to(device)
        
        # Create alignment model
        model = CEAlignmentInformation(
            x1_dim=x1_dim, x2_dim=x2_dim,
            hidden_dim=hidden_dim, embed_dim=embed_dim, num_labels=n_labels, 
            layers=layers, activation=activation,
            discrim_1=model_discrim_1, discrim_2=model_discrim_2, 
            discrim_12=model_discrim_12, p_y=p_y
        ).to(device)
        
        opt_align = optim.Adam(model.align_parameters(), lr=lr)
        
        # Train alignment
        train_loader = DataLoader(batch_train_ds, shuffle=True, 
                                batch_size=min(batch_size, len(batch_train_ds)),
                                num_workers=0)
        model.train()
        train_ce_alignment(model, train_loader, opt_align, 
                        data_type=([1], [2], [0]), num_epoch=ce_epochs, device=device)
        
        # Evaluate on test set
        model.eval()
        test_loader = DataLoader(test_ds, shuffle=False, 
                            batch_size=min(batch_size, len(test_ds)),
                            num_workers=0)
        results, _ = eval_ce_alignment(model, test_loader, data_type=([1], [2], [0]), device=device)
        
        # Average results across batches within this iteration
        batch_result = torch.mean(results, dim=0).cpu().numpy() / np.log(2)  # Convert to bits
        all_results.append(batch_result)
        
    # Average across all batches
    avg_results = np.mean(all_results, axis=0)

    return {
        'redundancy': max(0, avg_results[0]),
        'unique_x1': max(0, avg_results[1]),
        'unique_x2': max(0, avg_results[2]),
        'synergy': max(0, avg_results[3]),
        'total_di': sum(max(0, x) for x in avg_results),
        'method': 'batch'
    }
