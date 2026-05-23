import numpy as np
import cvxpy as cp
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import warnings
import pdb
from tqdm import tqdm

# Import batch estimation utilities
try:
    from estimators.ce_alignment_information import (
        CEAlignmentInformation, Discrim, MultimodalDataset,
        train_discrim, eval_discrim, train_ce_alignment, eval_ce_alignment
    )
    BATCH_AVAILABLE = True
except ImportError:
    warnings.warn("Batch estimation method not available. Install required dependencies.")
    BATCH_AVAILABLE = False

# Import multi-scale BATCH estimator
try:
    from estimators.multiscale_batch import (
        MultiScaleDiscrim, MultiScaleDiscrim12, MultiScaleCEAlignment,
        MultiLagDataset, train_multiscale_discriminators,
        train_multiscale_alignment, decompose_multilag_rus,
    )
    MULTISCALE_BATCH_AVAILABLE = True
except ImportError:
    warnings.warn("Multi-scale batch estimator not available.")
    MULTISCALE_BATCH_AVAILABLE = False

# Import base temporal PID functions
from pid.temporal_pid import (
    MI, CoI_temporal, UI_temporal, CI_temporal,
    estimate_transfer_entropy, plot_multi_lag_results
)


def reduce_dimensionality(X, n_components=10, method='pca'):
    """
    Reduce dimensionality of multivariate data.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data of shape (n_samples, n_features)
    n_components : int
        Number of components to keep
    method : str
        Dimensionality reduction method ('pca' or 'clustering')
        
    Returns:
    --------
    X_reduced : numpy.ndarray
        Reduced data
    reducer : object
        Fitted reducer for transforming new data
    """
    if X.shape[1] <= n_components:
        return X, None
        
    if method == 'pca':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        reducer = (scaler, pca)
    elif method == 'clustering':
        # Use K-means clustering to create discrete features
        kmeans = KMeans(n_clusters=n_components, random_state=42)
        X_reduced = kmeans.fit_predict(X).reshape(-1, 1)
        reducer = kmeans
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return X_reduced, reducer


def create_multivariate_probability_distribution(X1, X2, Y, lag=1, bins=10, 
                                               dim_reduction='auto', n_components=10):
    """
    Create joint probability distribution from multivariate time series.
    
    Parameters:
    -----------
    X1, X2 : numpy.ndarray
        Multivariate time series data (n_samples, n_features)
    Y : numpy.ndarray
        Target time series (n_samples,) or (n_samples, 1)
    lag : int
        Time lag
    bins : int
        Number of bins for discretization
    dim_reduction : str
        Dimensionality reduction method ('auto', 'pca', 'clustering', 'none')
    n_components : int
        Number of components for dimensionality reduction
        
    Returns:
    --------
    P : numpy.ndarray
        Joint probability distribution
    info : dict
        Additional information including reducers
    """
    # Handle multivariate inputs
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)
    if len(Y.shape) > 1:
        Y = Y.squeeze()
        
    # Adjust for lag
    if lag == 0:
        X1_past = X1
        X2_past = X2
        Y_present = Y
    else:
        X1_past = X1[:-lag]
        X2_past = X2[:-lag]
        Y_present = Y[lag:]
        
    # Apply dimensionality reduction if needed
    info = {'reducers': {}}
    
    if dim_reduction == 'auto':
        # Automatically decide based on dimensionality
        total_dims = X1_past.shape[1] + X2_past.shape[1]
        if total_dims > 20:
            dim_reduction = 'clustering'
        elif total_dims > 10:
            dim_reduction = 'pca'
        else:
            dim_reduction = 'none'
            
    if dim_reduction != 'none':
        X1_reduced, reducer1 = reduce_dimensionality(
            X1_past, n_components=min(n_components, X1_past.shape[1]), 
            method=dim_reduction
        )
        X2_reduced, reducer2 = reduce_dimensionality(
            X2_past, n_components=min(n_components, X2_past.shape[1]), 
            method=dim_reduction
        )
        info['reducers']['X1'] = reducer1
        info['reducers']['X2'] = reducer2
    else:
        X1_reduced = X1_past
        X2_reduced = X2_past
        
    # For clustering method, data is already discrete
    if dim_reduction == 'clustering':
        # Get unique values for each variable
        x1_vals = np.unique(X1_reduced)
        x2_vals = np.unique(X2_reduced)
        y_vals = np.unique(Y_present)
        
        n_x1 = len(x1_vals)
        n_x2 = len(x2_vals)
        n_y = len(y_vals)
        
        # Create mapping
        x1_map = {val: i for i, val in enumerate(x1_vals)}
        x2_map = {val: i for i, val in enumerate(x2_vals)}
        y_map = {val: i for i, val in enumerate(y_vals)}
        
        # Create joint distribution
        P = np.zeros((n_x1, n_x2, n_y))
        for i in range(len(Y_present)):
            x1_idx = x1_map[X1_reduced[i, 0]]
            x2_idx = x2_map[X2_reduced[i, 0]]
            y_idx = y_map[Y_present[i]]
            P[x1_idx, x2_idx, y_idx] += 1
            
    else:
        # Discretize continuous data
        # Flatten multivariate data for binning
        X1_flat = X1_reduced.flatten()
        X2_flat = X2_reduced.flatten()
        
        # Create bins
        x1_edges = np.linspace(X1_flat.min(), X1_flat.max(), bins + 1)
        x2_edges = np.linspace(X2_flat.min(), X2_flat.max(), bins + 1)
        y_edges = np.linspace(Y_present.min(), Y_present.max(), bins + 1)
        
        # For multivariate data, we need to bin each sample
        P = np.zeros((bins, bins, bins))
        
        for i in range(len(Y_present)):
            # Use first principal component or average for binning
            if X1_reduced.shape[1] > 1:
                x1_val = np.mean(X1_reduced[i])
            else:
                x1_val = X1_reduced[i, 0]
                
            if X2_reduced.shape[1] > 1:
                x2_val = np.mean(X2_reduced[i])
            else:
                x2_val = X2_reduced[i, 0]
                
            x1_bin = np.clip(np.digitize(x1_val, x1_edges) - 1, 0, bins - 1)
            x2_bin = np.clip(np.digitize(x2_val, x2_edges) - 1, 0, bins - 1)
            y_bin = np.clip(np.digitize(Y_present[i], y_edges) - 1, 0, bins - 1)
            
            P[x1_bin, x2_bin, y_bin] += 1
            
    # Normalize
    P = P / (np.sum(P) + 1e-10)
    
    info['shape'] = P.shape
    return P, info


def solve_Q_cvxpy_regularized(P: np.ndarray, regularization=1e-6, verbose=False):
    """
    Solve for Q using CVXPY with regularization for numerical stability.
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D joint probability distribution
    regularization : float
        Regularization parameter for numerical stability
    verbose : bool
        Whether to print optimization details
        
    Returns:
    --------
    Q : numpy.ndarray
        Optimized distribution
    """
    # Add small regularization to P to ensure positive values
    P = P + regularization
    P = P / np.sum(P)
    
    # Compute marginals
    Px1y = P.sum(axis=1)
    Px2y = P.sum(axis=0)
    
    # Define optimization variables
    Q = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) 
         for _ in range(P.shape[2])]
    Q_x1x2 = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) 
              for _ in range(P.shape[2])]
    
    # Constraints
    constraints = []
    
    # Sum to one constraint
    constraints.append(cp.sum([cp.sum(q) for q in Q]) == 1)
    
    # Marginal constraints
    # P(X1, Y) = Q(X1, Y)
    for x1 in range(P.shape[0]):
        for y in range(P.shape[2]):
            constraints.append(
                cp.sum([Q[y][x1, x2] for x2 in range(P.shape[1])]) == Px1y[x1, y]
            )
    
    # P(X2, Y) = Q(X2, Y)
    for x2 in range(P.shape[1]):
        for y in range(P.shape[2]):
            constraints.append(
                cp.sum([Q[y][x1, x2] for x1 in range(P.shape[0])]) == Px2y[x2, y]
            )
    
    # Product distribution constraints
    for i in range(P.shape[2]):
        constraints.append(cp.sum(Q) / P.shape[2] == Q_x1x2[i])
    
    # Objective: minimize I(X1; X2 | Y)
    obj = cp.sum([cp.sum(cp.rel_entr(Q[i], Q_x1x2[i])) 
                  for i in range(P.shape[2])])
    
    # Add regularization term to objective for stability
    reg_term = regularization * cp.sum([cp.sum(cp.square(q)) for q in Q])
    
    prob = cp.Problem(cp.Minimize(obj + reg_term), constraints)
    
    # Solve with multiple solvers if needed
    try:
        prob.solve(verbose=verbose, max_iters=50000, solver=cp.CLARABEL)
    except:
        try:
            prob.solve(verbose=verbose, max_iters=50000, solver=cp.SCS)
        except:
            prob.solve(verbose=verbose, max_iters=50000)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        warnings.warn(f"Optimization problem status: {prob.status}")
        
    # Extract solution
    Q_solution = np.stack([q.value for q in Q], axis=2)
    
    # Ensure non-negative and normalized
    Q_solution = np.maximum(Q_solution, 0)
    Q_solution = Q_solution / (np.sum(Q_solution) + 1e-10)
    
    return Q_solution


def temporal_pid_batch(X1, X2, Y, lag=1, batch_size=256, n_batches=10, 
                      discrim_epochs=20, ce_epochs=10, seed=42, device=None,
                      hidden_dim=32, layers=2, activation='relu', lr=1e-3, embed_dim=10):
    """
    Compute PID using batch/neural network method for high-dimensional data.
    
    Parameters:
    -----------
    X1, X2 : numpy.ndarray
        Multivariate time series (n_samples, n_features)
    Y : numpy.ndarray
        Target time series
    lag : int
        Time lag
    batch_size : int
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
        
    Returns:
    --------
    results : dict
        PID components
    """
    if not BATCH_AVAILABLE:
        raise RuntimeError("Batch method not available. Please install required dependencies.")
        
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize inputs (see temporal_pid_multiscale_batch for rationale).
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)
    X1 = (X1 - X1.mean(axis=0, keepdims=True)) / (X1.std(axis=0, keepdims=True) + 1e-8)
    X2 = (X2 - X2.mean(axis=0, keepdims=True)) / (X2.std(axis=0, keepdims=True) + 1e-8)

    # Prepare lagged data
    if lag > 0:
        X1_past = X1[:-lag]
        X2_past = X2[:-lag]
        Y_present = Y[lag:]
    else:
        X1_past = X1
        X2_past = X2
        Y_present = Y
        
    # Convert to tensors
    X1_tensor = torch.tensor(X1_past, dtype=torch.float32).to(device)
    X2_tensor = torch.tensor(X2_past, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y_present, dtype=torch.long).to(device)
    
    # Handle multiclass targets
    num_labels = len(np.unique(Y_present))

    # Create train/test split - use temporal split to preserve time structure
    n_samples = len(Y_present)
    n_train = int(0.8 * n_samples)

    # Use temporal split instead of random permutation to preserve temporal structure
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_samples)
    
    # Create datasets
    train_ds = MultimodalDataset(
        [X1_tensor[train_idx], X2_tensor[train_idx]], 
        Y_tensor[train_idx]
    )
    test_ds = MultimodalDataset(
        [X1_tensor[test_idx], X2_tensor[test_idx]], 
        Y_tensor[test_idx]
    )
    
    # Run multiple batches and average results
    all_results = []
    # TODO: do we need to do this batch-wise? and then average?
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
        
        model_discrim_1 = Discrim(x_dim=x1_dim, hidden_dim=hidden_dim, num_labels=num_labels, 
                                 layers=layers, activation=activation).to(device)
        model_discrim_2 = Discrim(x_dim=x2_dim, hidden_dim=hidden_dim, num_labels=num_labels, 
                                 layers=layers, activation=activation).to(device)
        model_discrim_12 = Discrim(x_dim=x1_dim + x2_dim, hidden_dim=hidden_dim, 
                                  num_labels=num_labels, layers=layers, activation=activation).to(device)
        
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
            
        # Compute prior P(Y) from the full lagged series so temporal split does
        # not leave rare activities with p_y=0, which causes log(0)=-inf and
        # propagates nan through all MI terms (yielding all-zero RUS values).
        p_y = torch.zeros(num_labels)
        for i in range(num_labels):
            p_y[i] = (Y_tensor == i).float().mean()
        p_y = p_y.to(device)
        
        # Create alignment model
        model = CEAlignmentInformation(
            x1_dim=x1_dim, x2_dim=x2_dim,
            hidden_dim=hidden_dim, embed_dim=embed_dim, num_labels=num_labels, 
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
        
        # Evaluate
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


def temporal_pid_multiscale_batch(X1, X2, Y, max_lag=5, batch_size=256,
                                  discrim_epochs=20, ce_epochs=10,
                                  steps_per_epoch=None, seed=42, device=None,
                                  hidden_dim=32, layers=2, activation='relu',
                                  lr=1e-3, embed_dim=10, lag_embed_dim=8,
                                  lag_weights=None, **kwargs):
    """Multi-scale BATCH estimator: one model that returns RUS for every lag.

    Trains a single set of lag-conditioned discriminators and a single
    alignment module across all lags in {0, ..., max_lag}, then reads out the
    RUS decomposition for each lag from the trained model. Replaces the per-lag
    BATCH loop with a single multi-task training run; see Algorithms 1-3 in
    Appendix C of the MERGE paper.

    Returns a dict in the same shape as multi_lag_analysis: lists indexed by
    lag for redundancy / unique_x1 / unique_x2 / synergy / total_di.
    """
    if not MULTISCALE_BATCH_AVAILABLE:
        raise RuntimeError("Multi-scale batch estimator unavailable.")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)
    if len(Y.shape) > 1:
        Y = Y.squeeze()

    if len(Y) <= max_lag + 1:
        raise ValueError(
            f"Series length {len(Y)} too short for max_lag {max_lag}.")

    # Standardize each modality to zero mean / unit variance so that the encoder
    # networks receive well-conditioned inputs regardless of sensor scales.
    # Without this, PAMAP sensors (temperature ~35°C, magnetometer ~50µT, etc.)
    # cause dead ReLUs and near-zero embedding variance, which overflows exp() in
    # the Sinkhorn alignment and collapses all RUS values to 0 / inf.
    X1_mean = X1.mean(axis=0, keepdims=True)
    X1_std  = X1.std(axis=0, keepdims=True) + 1e-8
    X2_mean = X2.mean(axis=0, keepdims=True)
    X2_std  = X2.std(axis=0, keepdims=True) + 1e-8
    X1 = (X1 - X1_mean) / X1_std
    X2 = (X2 - X2_mean) / X2_std

    X1_t = torch.as_tensor(X1, dtype=torch.float32)
    X2_t = torch.as_tensor(X2, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.long)

    num_labels = int(Y_t.max().item()) + 1
    num_lags = max_lag + 1

    data = MultiLagDataset(X1_t, X2_t, Y_t, max_lag=max_lag, device=device)

    # Default steps_per_epoch: cover the smallest lagged dataset ~once per
    # epoch in expectation when sampling lags uniformly.
    if steps_per_epoch is None:
        smallest = min(data.size(tau) for tau in range(num_lags))
        steps_per_epoch = max(1, smallest // batch_size)

    x1_dim = X1_t.shape[1]
    x2_dim = X2_t.shape[1]

    discrim1 = MultiScaleDiscrim(x1_dim, hidden_dim, num_labels, layers,
                                 activation, lag_embed_dim, num_lags).to(device)
    discrim2 = MultiScaleDiscrim(x2_dim, hidden_dim, num_labels, layers,
                                 activation, lag_embed_dim, num_lags).to(device)
    discrim12 = MultiScaleDiscrim12(x1_dim, x2_dim, hidden_dim, num_labels,
                                    layers, activation, lag_embed_dim,
                                    num_lags).to(device)

    train_multiscale_discriminators(
        discrim1, discrim2, discrim12, data,
        num_epochs=discrim_epochs, steps_per_epoch=steps_per_epoch,
        batch_size=batch_size, lr=lr, device=device, lag_weights=lag_weights,
    )

    align_model = MultiScaleCEAlignment(
        x1_dim=x1_dim, x2_dim=x2_dim, hidden_dim=hidden_dim,
        embed_dim=embed_dim, num_labels=num_labels, layers=layers,
        activation=activation, lag_embed_dim=lag_embed_dim, num_lags=num_lags,
    ).to(device)

    train_multiscale_alignment(
        align_model, discrim1, discrim2, data,
        num_epochs=ce_epochs, steps_per_epoch=steps_per_epoch,
        batch_size=batch_size, lr=lr, device=device, lag_weights=lag_weights,
    )

    # Empirical P(Y) across the whole series (lag-invariant up to boundary).
    p_y = torch.zeros(num_labels, device=device)
    for i in range(num_labels):
        p_y[i] = (Y_t == i).float().mean()

    per_lag = decompose_multilag_rus(
        align_model, discrim1, discrim2, discrim12, data, p_y, device,
        batch_size,
    )

    return {
        'lag': [r['lag'] for r in per_lag],
        'redundancy': [r['redundancy'] for r in per_lag],
        'unique_x1': [r['unique_x1'] for r in per_lag],
        'unique_x2': [r['unique_x2'] for r in per_lag],
        'synergy': [r['synergy'] for r in per_lag],
        'total_di': [r['total_di'] for r in per_lag],
        'method': ['multiscale_batch'] * num_lags,
    }


def temporal_pid_multivariate(X1, X2, Y, lag=1, bins=10, method='auto',
                            dim_reduction='auto', n_components=10,
                            regularization=1e-6, batch_size=256, 
                            n_batches=10, seed=42, device=None,
                            hidden_dim=32, layers=2, activation='relu', lr=1e-3, 
                            embed_dim=10, discrim_epochs=20, ce_epochs=10, **kwargs):
    """
    Compute temporal PID for multivariate time series.
    
    Parameters:
    -----------
    X1, X2 : numpy.ndarray
        Multivariate time series data
    Y : numpy.ndarray
        Target time series
    lag : int
        Time lag
    bins : int
        Number of bins for discretization
    method : str
        PID estimation method ('auto', 'joint', 'cvxpy', 'batch')
    dim_reduction : str
        Dimensionality reduction method
    n_components : int
        Number of components for dim reduction
    regularization : float
        Regularization for CVXPY method
    batch_size : int
        Batch size for batch method
    n_batches : int
        Number of batches for batch method
    seed : int
        Random seed
    device : torch.device
        Device to run computations on for batch method
    hidden_dim : int
        Hidden dimension for neural networks in batch method
    layers : int
        Number of layers for neural networks in batch method
    activation : str
        Activation function for neural networks in batch method
    lr : float
        Learning rate for neural networks in batch method
    embed_dim : int
        Embedding dimension for alignment model in batch method
    discrim_epochs : int
        Number of epochs for discriminator training in batch method
    ce_epochs : int
        Number of epochs for CE alignment training in batch method
    **kwargs : dict
        Additional method-specific parameters
        
    Returns:
    --------
    results : dict
        PID components
    """
    # Auto-select method based on dimensionality
    if method == 'auto':
        total_features = (X1.shape[1] if len(X1.shape) > 1 else 1) + \
                        (X2.shape[1] if len(X2.shape) > 1 else 1)
        
        if total_features <= 10:
            method = 'joint'
        elif total_features <= 50:
            method = 'cvxpy'
        else:
            method = 'batch'
            
    print(f"Using {method} method for PID estimation")
    
    if method == 'batch':
        # Use neural network-based method
        return temporal_pid_batch(
            X1, X2, Y, lag=lag, batch_size=batch_size, 
            n_batches=n_batches, seed=seed, device=device,
            hidden_dim=hidden_dim, layers=layers, activation=activation,
            lr=lr, embed_dim=embed_dim, discrim_epochs=discrim_epochs,
            ce_epochs=ce_epochs, **kwargs
        )
    
    else:
        # Use histogram-based methods (joint or cvxpy)
        # Create probability distribution
        P, info = create_multivariate_probability_distribution(
            X1, X2, Y, lag=lag, bins=bins, 
            dim_reduction=dim_reduction, n_components=n_components
        )
        
        if method == 'cvxpy':
            # Use CVXPY optimization
            Q = solve_Q_cvxpy_regularized(P, regularization=regularization)
        else:
            # Use standard optimization (joint method)
            from temporal_pid import solve_Q_temporal
            Q = solve_Q_temporal(P)
            
        # Calculate PID components
        redundancy = CoI_temporal(Q)
        unique_x1 = UI_temporal(Q, cond_id=1)
        unique_x2 = UI_temporal(Q, cond_id=0)
        synergy = CI_temporal(P, Q)
        
        # Calculate total directed information
        P_reshaped = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))
        total_di = MI(P_reshaped)
        
        return {
            'redundancy': redundancy,
            'unique_x1': unique_x1,
            'unique_x2': unique_x2,
            'synergy': synergy,
            'total_di': total_di,
            'sum_components': redundancy + unique_x1 + unique_x2 + synergy,
            'method': method,
            'info': info
        }


def multi_lag_analysis(X1, X2, Y, max_lag=5, bins=10, method='auto', **kwargs):
    """
    Perform PID analysis across multiple time lags with support for different methods.
    
    Parameters:
    -----------
    X1, X2, Y : numpy.ndarray
        Time series data (can be multivariate for X1, X2)
    max_lag : int
        Maximum time lag to consider
    bins : int
        Number of bins for discretization
    method : str
        PID estimation method ('auto', 'joint', 'cvxpy', 'batch',
        'multiscale_batch'). 'multiscale_batch' trains a single
        lag-conditioned model across all lags in one shot.
    **kwargs : dict
        Additional method-specific parameters
        
    Returns:
    --------
    results : dict
        Dictionary containing PID components for each lag
    """
    # Multi-scale BATCH bypasses the per-lag loop entirely: one model trained
    # across all lags returns the full RUS sequence directly.
    if method == 'multiscale_batch':
        return temporal_pid_multiscale_batch(X1, X2, Y, max_lag=max_lag,
                                             **kwargs)

    # Check if data is multivariate
    is_multivariate = (len(X1.shape) > 1 and X1.shape[1] > 1) or \
                     (len(X2.shape) > 1 and X2.shape[1] > 1)

    results = {
        'lag': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_di': [],
        'method': []
    }

    for lag in tqdm(range(max_lag + 1), desc="Processing lags"):
        
        # try:
        if is_multivariate:
            # Use multivariate version
            pid_result = temporal_pid_multivariate(
                X1, X2, Y, lag=lag, bins=bins, method=method, **kwargs
            )
        else:
            # Use standard version for univariate data
            from temporal_pid import temporal_pid
            if method in ['joint', 'auto']:
                pid_result = temporal_pid(X1, X2, Y, lag=lag, bins=bins)
            else:
                # Fall back to multivariate version for other methods
                pid_result = temporal_pid_multivariate(
                    X1, X2, Y, lag=lag, bins=bins, method=method, **kwargs
                )
        
        results['lag'].append(lag)
        results['redundancy'].append(pid_result['redundancy'])
        results['unique_x1'].append(pid_result['unique_x1'])
        results['unique_x2'].append(pid_result['unique_x2'])
        results['synergy'].append(pid_result['synergy'])
        results['total_di'].append(pid_result['total_di'])
        results['method'].append(pid_result.get('method', method))
            
        # except Exception as e:
        #     print(f"Error at lag {lag}: {str(e)}")
        #     # Append NaN values for failed lags
        #     results['lag'].append(lag)
        #     for key in ['redundancy', 'unique_x1', 'unique_x2', 'synergy', 'total_di']:
        #         results[key].append(np.nan)
        #     results['method'].append('failed')
    
    return results


# Example usage
if __name__ == "__main__":
    print("Testing multivariate temporal PID...")
    
    # Generate synthetic multivariate data
    n_samples = 1000
    n_features1 = 5
    n_features2 = 8
    
    # Create correlated multivariate time series
    X1 = np.random.randn(n_samples, n_features1)
    X2 = np.random.randn(n_samples, n_features2)
    
    # Create target with influences from both sources
    Y = np.zeros(n_samples)
    for t in range(2, n_samples):
        # Influence from X1 (using first two features)
        Y[t] += 0.5 * X1[t-1, 0] + 0.3 * X1[t-1, 1]
        # Influence from X2 (using first feature)
        Y[t] += 0.4 * X2[t-2, 0]
        # Synergistic effect
        Y[t] += 0.2 * X1[t-1, 0] * X2[t-2, 0]
        # Noise
        Y[t] += 0.1 * np.random.randn()
    
    # Discretize Y for classification
    Y_discrete = (Y > np.median(Y)).astype(int)
    
    print(f"Data shapes: X1={X1.shape}, X2={X2.shape}, Y={Y_discrete.shape}")
    
    # Test different methods
    for method in ['joint', 'cvxpy', 'batch', 'multiscale_batch']:
        print(f"\n{'='*50}")
        print(f"Testing {method} method...")
        
        try:
            results = multi_lag_analysis(
                X1, X2, Y_discrete, 
                max_lag=3, 
                bins=4, 
                method=method,
                dim_reduction='pca',
                n_components=3,
                batch_size=100,
                n_batches=3,
                discrim_epochs=5,
                ce_epochs=5
            )
            
            print("\nResults:")
            for lag in range(len(results['lag'])):
                print(f"Lag {results['lag'][lag]}: "
                      f"R={results['redundancy'][lag]:.3f}, "
                      f"U1={results['unique_x1'][lag]:.3f}, "
                      f"U2={results['unique_x2'][lag]:.3f}, "
                      f"S={results['synergy'][lag]:.3f}, "
                      f"Total={results['total_di'][lag]:.3f}")
                      
        except Exception as e:
            print(f"Method {method} failed: {str(e)}") 