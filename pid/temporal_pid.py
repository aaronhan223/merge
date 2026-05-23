import numpy as np
import cvxpy as cp
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import os
import pdb

def estimate_transfer_entropy(source, target, lag=1, bins=10, method='binned'):
    """
    Estimate transfer entropy from source to target with specified lag.
    
    Transfer entropy measures the amount of directed information flow from 
    source to target, accounting for the history of the target.
    
    TE(source→target) = I(target_present; source_past | target_past)
    
    Parameters:
    -----------
    source : numpy.ndarray
        Source time series
    target : numpy.ndarray
        Target time series
    lag : int, default=1
        Time lag to consider for causal influence
    bins : int, default=10
        Number of bins for discretization (only used if method='binned')
    method : str, default='binned'
        Method for estimation: 'binned' or 'ksg' (KSG estimator)
        
    Returns:
    --------
    te : float
        Transfer entropy in bits
    """
    if len(source) != len(target):
        raise ValueError("Source and target must have the same length")
        
    # Create lagged variables
    source_past = source[:-lag]
    target_past = target[:-lag]
    target_present = target[lag:]
    # Discretize data if continuous
    if not np.array_equal(source, source.astype(int)) or not np.array_equal(target, target.astype(int)):
        # Discretize continuous data using digitize instead of histogram
        source_bins = np.linspace(min(source_past), max(source_past), bins+1)
        target_bins = np.linspace(min(target), max(target), bins+1)
        
        source_past_disc = np.digitize(source_past, source_bins)
        target_past_disc = np.digitize(target_past, target_bins)
        target_present_disc = np.digitize(target_present, target_bins)
    else:
        # Already discrete
        source_past_disc = source_past
        target_past_disc = target_past
        target_present_disc = target_present
    
    # Calculate transfer entropy using rel_entr from scipy.special
    if bins <= 5:  # Only use this approach for reasonable bin sizes
        # Create joint probability distributions
        
        # 1. Joint distribution P(target_present, source_past, target_past)
        P_joint = np.zeros((bins, bins, bins))
        # 2. Joint distribution P(target_present, target_past)
        P_target = np.zeros((bins, bins))
        
        # Count occurrences to estimate probabilities
        for i in range(len(target_present_disc)):
            # Ensure indices are within bounds
            tp = min(target_present_disc[i]-1, bins-1)
            sp = min(source_past_disc[i]-1, bins-1)
            tpa = min(target_past_disc[i]-1, bins-1)
            
            # Handle case where indices are negative (below the first bin edge)
            tp = max(tp, 0)
            sp = max(sp, 0)
            tpa = max(tpa, 0)
            
            # Count joint occurrences
            P_joint[tp, sp, tpa] += 1
            P_target[tp, tpa] += 1
        
        # Normalize to get probability distributions
        P_joint = P_joint / np.sum(P_joint)
        P_target = P_target / np.sum(P_target)
        
        # Compute conditional distributions
        
        # 1. P(target_present | source_past, target_past)
        P_joint_cond = np.zeros_like(P_joint)
        P_source_target_past = np.sum(P_joint, axis=0)  # P(source_past, target_past)
        
        for tp in range(bins):
            for sp in range(bins):
                for tpa in range(bins):
                    if P_source_target_past[sp, tpa] > 0:
                        P_joint_cond[tp, sp, tpa] = P_joint[tp, sp, tpa] / P_source_target_past[sp, tpa]
        
        # 2. P(target_present | target_past)
        P_target_cond = np.zeros_like(P_target)
        P_target_past = np.sum(P_target, axis=0)  # P(target_past)
        
        for tp in range(bins):
            for tpa in range(bins):
                if P_target_past[tpa] > 0:
                    P_target_cond[tp, tpa] = P_target[tp, tpa] / P_target_past[tpa]
        
        # Calculate KL divergence (relative entropy) between conditional distributions using rel_entr
        te = 0.0
        for tp in range(bins):
            for sp in range(bins):
                for tpa in range(bins):
                    if P_joint[tp, sp, tpa] > 0 and P_target_cond[tp, tpa] > 0:
                        # Use rel_entr from scipy.special
                        # rel_entr(p, q) = p * log(p / q)
                        te += rel_entr(P_joint_cond[tp, sp, tpa], P_target_cond[tp, tpa]) * P_joint[tp, sp, tpa]
        
        # Convert from nats to bits (rel_entr uses natural log)
        te = te / np.log(2)
        return max(0, te)  # Ensure non-negative
    else:
        # For larger bin sizes, fall back to the original implementation which is more efficient
        # Create joint variable by encoding the joint state as a single integer
        joint_past_disc = source_past_disc * bins + target_past_disc
        
        # Calculate I(target_present; source_past, target_past)
        mi_joint = mutual_info_score(joint_past_disc, target_present_disc) / np.log(2)  # Convert to bits
        
        # Calculate I(target_present; target_past)
        mi_target_past = mutual_info_score(target_past_disc, target_present_disc) / np.log(2)  # Convert to bits
        
        # Transfer entropy
        te = mi_joint - mi_target_past
        return max(0, te)  # Ensure non-negative
        

def create_probability_distribution(X1, X2, Y, lag=1, bins=10):
    """
    Create a joint probability distribution from time series data with lag.
    
    Parameters:
    -----------
    X1, X2, Y : numpy.ndarray
        Time series data
    lag : int, default=1
        Time lag to consider for causal influence
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    P : numpy.ndarray
        3D array of joint probability distribution P(X1_past, X2_past, Y_present)
    """
    # Adjust for lag
    if lag == 0:
        X1_past = X1
        X2_past = X2
        Y_present = Y
    else:
        X1_past = X1[:-lag]
        X2_past = X2[:-lag]
        Y_present = Y[lag:]
    # Discretize continuous data if needed
    if not np.array_equal(X1_past, X1_past.astype(int)) or not np.array_equal(X2_past, X2_past.astype(int)) or not np.array_equal(Y_present, Y_present.astype(int)):
        # Get bin edges
        x1_edges = np.linspace(min(X1_past), max(X1_past), bins+1)
        x2_edges = np.linspace(min(X2_past), max(X2_past), bins+1)
        y_edges = np.linspace(min(Y_present), max(Y_present), bins+1)

        # Discretize
        x1_bins = np.digitize(X1_past, x1_edges) - 1
        x2_bins = np.digitize(X2_past, x2_edges) - 1
        y_bins = np.digitize(Y_present, y_edges) - 1

        # Ensure values are within bins range
        x1_bins = np.clip(x1_bins, 0, bins-1)
        x2_bins = np.clip(x2_bins, 0, bins-1)
        y_bins = np.clip(y_bins, 0, bins-1)
    else:
        # Already discrete
        x1_bins = X1_past
        x2_bins = X2_past
        y_bins = Y_present
        
        # Get unique values
        x1_unique = np.unique(x1_bins)
        x2_unique = np.unique(x2_bins)
        y_unique = np.unique(y_bins)
        
        # Remap to contiguous integers
        x1_map = {val: i for i, val in enumerate(x1_unique)}
        x2_map = {val: i for i, val in enumerate(x2_unique)}
        y_map = {val: i for i, val in enumerate(y_unique)}
        
        x1_bins = np.array([x1_map[val] for val in x1_bins])
        x2_bins = np.array([x2_map[val] for val in x2_bins])
        y_bins = np.array([y_map[val] for val in y_bins])
        
        bins = max(len(x1_unique), len(x2_unique), len(y_unique))

    # Create histogram
    P = np.zeros((bins, bins, bins))
    for i in range(len(x1_bins)):
        P[x1_bins[i], x2_bins[i], y_bins[i]] += 1

    # Normalize
    P = P / np.sum(P)

    return P

def MI(P: np.ndarray):
    """
    Calculate mutual information from a 2D joint probability distribution.
    """
    margin_1 = P.sum(axis=1)
    margin_2 = P.sum(axis=0)
    outer = np.outer(margin_1, margin_2)
    
    # Calculate KL divergence
    return np.sum(rel_entr(P, outer))

def solve_Q_temporal(P: np.ndarray):
    """
    Compute optimal Q given 3D array P with temporal consideration.
    
    This function solves an optimization problem to find the distribution Q
    that preserves marginals P(X1,Y) and P(X2,Y) while minimizing I(X1;X2|Y).
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D joint probability distribution P(X1_past, X2_past, Y_present)
        
    Returns:
    --------
    Q : numpy.ndarray
        Optimized joint distribution with minimal synergy
    """
    # Compute marginals
    Py = P.sum(axis=0).sum(axis=0)
    Px1 = P.sum(axis=1).sum(axis=1)
    Px2 = P.sum(axis=0).sum(axis=1)
    Px2y = P.sum(axis=0)
    Px1y = P.sum(axis=1)
    
    # Define optimization variables
    Q = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) for i in range(P.shape[2])]
    Q_x1x2 = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) for i in range(P.shape[2])]

    # Constraints that conditional distributions sum to 1
    sum_to_one_Q = cp.sum([cp.sum(q) for q in Q]) == 1

    # [A]: p(x1, y) == q(x1, y) constraints
    A_cstrs = []
    for x1 in range(P.shape[0]):
        for y in range(P.shape[2]):
            vars = []
            for x2 in range(P.shape[1]):
                vars.append(Q[y][x1, x2])
            A_cstrs.append(cp.sum(vars) == Px1y[x1,y])
    
    # [B]: p(x2, y) == q(x2, y) constraints
    B_cstrs = []
    for x2 in range(P.shape[1]):
        for y in range(P.shape[2]):
            vars = []
            for x1 in range(P.shape[0]):
                vars.append(Q[y][x1, x2])
            B_cstrs.append(cp.sum(vars) == Px2y[x2,y])

    # KL divergence - Product distribution constraints
    Q_pdt_dist_cstrs = [cp.sum(Q) / P.shape[2] == Q_x1x2[i] for i in range(P.shape[2])]

    # Objective: minimize I(X1; X2 | Y)
    obj = cp.sum([cp.sum(cp.rel_entr(Q[i], Q_x1x2[i])) for i in range(P.shape[2])])
    
    all_constrs = []
    all_constrs.append(sum_to_one_Q)
    all_constrs.extend(A_cstrs)
    all_constrs.extend(B_cstrs) 
    all_constrs.extend(Q_pdt_dist_cstrs)
    
    prob = cp.Problem(cp.Minimize(obj), all_constrs)
    
    # Solve with better error handling
    try:
        prob.solve(verbose=False, max_iter=50000)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem status is {prob.status}")
    except Exception as e:
        print(f"Optimization error: {e}")
        # Try with different solver
        try:
            prob.solve(verbose=False, max_iter=50000, solver=cp.ECOS)
        except:
            print("Falling back to SCS solver")
            prob.solve(verbose=False, max_iter=50000, solver=cp.SCS)

    # Convert to numpy array
    Q_solution = []
    for q in Q:
        if q.value is not None:
            Q_solution.append(q.value)
        else:
            # If optimization failed, return uniform distribution
            Q_solution.append(np.ones((P.shape[0], P.shape[1])) / (P.shape[0] * P.shape[1]))
    
    return np.stack(Q_solution, axis=2)

def CoI_temporal(P: np.ndarray):
    """
    Calculate co-information (redundancy) from temporal distribution.
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D joint probability distribution P(X1_past, X2_past, Y_present)
        
    Returns:
    --------
    redundancy : float
        Redundant information between X1_past and X2_past about Y_present
    """
    # MI(Y; X1)
    A = P.sum(axis=1)

    # MI(Y; X2)
    B = P.sum(axis=0)

    # MI(Y; (X1, X2))
    C = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))
    
    # I(Y; X1; X2)
    return MI(A) + MI(B) - MI(C)

def UI_temporal(P, cond_id=0):
    """
    Calculate unique information from temporal distribution.
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D joint probability distribution P(X1_past, X2_past, Y_present)
    cond_id : int, default=0
        If 0, calculate unique information of X2; if 1, calculate unique information of X1
        
    Returns:
    --------
    unique_info : float
        Unique information from one source about Y_present
    """
    sum_val = 0.0

    if cond_id == 0:
        # Unique info from X2 (condition on X1)
        J = P.sum(axis=(1, 2))  # marginal of X1
        for i in range(P.shape[0]):
            P_slice = P[i,:,:]
            if np.sum(P_slice) > 0:  # Avoid division by zero
                sum_val += MI(P_slice/np.sum(P_slice)) * J[i]
    elif cond_id == 1:
        # Unique info from X1 (condition on X2)
        J = P.sum(axis=(0, 2))  # marginal of X2
        for i in range(P.shape[1]):
            P_slice = P[:,i,:]
            if np.sum(P_slice) > 0:  # Avoid division by zero
                sum_val += MI(P_slice/np.sum(P_slice)) * J[i]
    else:
        raise ValueError("cond_id must be 0 or 1")

    return sum_val

def CI_temporal(P, Q):
    """
    Calculate synergistic information from temporal distributions.
    
    Parameters:
    -----------
    P : numpy.ndarray
        Original 3D joint probability distribution P(X1_past, X2_past, Y_present)
    Q : numpy.ndarray
        Optimized 3D joint distribution with minimal synergy
        
    Returns:
    --------
    synergy : float
        Synergistic information from X1_past and X2_past about Y_present
    """
    # Ensure P and Q have the same shape
    assert P.shape == Q.shape
    
    # Reshape to 2D for mutual information calculation
    P_ = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))
    Q_ = Q.transpose([2, 0, 1]).reshape((Q.shape[2], Q.shape[0]*Q.shape[1]))
    
    # Calculate total MI in P minus total MI in Q (synergy)
    return MI(P_) - MI(Q_)

def temporal_pid(X1, X2, Y, lag=1, bins=10):
    """
    Compute partial information decomposition using directed information.
    
    This function decomposes the directed information from past X1, X2 to present Y
    into redundant, unique, and synergistic components.
    
    Parameters:
    -----------
    X1, X2, Y : numpy.ndarray
        Time series data
    lag : int, default=1
        Time lag to consider for causal influence
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    results : dict
        Dictionary containing redundancy, unique_x1, unique_x2, synergy values
    """
    # Create joint probability distribution
    P = create_probability_distribution(X1, X2, Y, lag, bins)

    # Optimize to get Q (distribution with minimal synergy)
    Q = solve_Q_temporal(P)
    
    # Calculate PID components
    redundancy = CoI_temporal(Q)
    unique_x1 = UI_temporal(Q, cond_id=1)
    unique_x2 = UI_temporal(Q, cond_id=0)
    synergy = CI_temporal(P, Q)
    
    # Calculate total directed information (should equal sum of components)
    total_di = MI(P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1])))
    
    # Create results dictionary
    results = {
        'redundancy': redundancy,
        'unique_x1': unique_x1,
        'unique_x2': unique_x2,
        'synergy': synergy,
        'total_di': total_di,
        'sum_components': redundancy + unique_x1 + unique_x2 + synergy
    }
    
    return results

def multi_lag_analysis(X1, X2, Y, max_lag=5, bins=10):
    """
    Perform PID analysis across multiple time lags.
    
    Parameters:
    -----------
    X1, X2, Y : numpy.ndarray
        Time series data (can be multivariate for X1, X2)
    max_lag : int, default=5
        Maximum time lag to consider
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    results : dict
        Dictionary containing PID components for each lag
    """
    results = {
        'lag': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_di': []
    }
    
    for lag in range(max_lag + 1):
        print(f"Analyzing lag {lag}...")
        pid_result = temporal_pid(X1, X2, Y, lag, bins)
        
        results['lag'].append(lag)
        results['redundancy'].append(pid_result['redundancy'])
        results['unique_x1'].append(pid_result['unique_x1'])
        results['unique_x2'].append(pid_result['unique_x2'])
        results['synergy'].append(pid_result['synergy'])
        results['total_di'].append(pid_result['total_di'])
    
    return results

def plot_multi_lag_results(results, title=None, save_path=None):
    """
    Plot PID components across multiple time lags.
    
    Parameters:
    -----------
    results : dict
        Output from multi_lag_analysis
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results['lag'], results['total_di'], 'ko-', linewidth=2, label='Total DI')
    plt.plot(results['lag'], results['redundancy'], 'b.-', label='Redundancy')
    plt.plot(results['lag'], results['unique_x1'], 'g.-', label='Unique X1')
    plt.plot(results['lag'], results['unique_x2'], 'r.-', label='Unique X2')
    plt.plot(results['lag'], results['synergy'], 'm.-', label='Synergy')
    plt.xlabel('Time Lag', fontsize=20)
    plt.ylabel('Information (bits)', fontsize=20)
    if title:
        plt.title(title, fontsize=24)
    else:
        plt.title('PID Components vs Time Lag', fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Stacked area plot
    plt.subplot(2, 1, 2)
    plt.stackplot(
        results['lag'], 
        [results['redundancy'], results['unique_x1'], results['unique_x2'], results['synergy']],
        labels=['Redundancy', 'Unique X1', 'Unique X2', 'Synergy'],
        colors=['blue', 'green', 'red', 'magenta'],
        alpha=0.7
    )
    plt.plot(results['lag'], results['total_di'], 'k--', linewidth=2, label='Total DI')
    plt.xlabel('Time Lag', fontsize=20)
    plt.ylabel('Information (bits)', fontsize=20)
    if title:
        plt.title(title, fontsize=24)
    else:
        plt.title('Stacked PID Components', fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def generate_causal_time_series(n_samples=1000, causal_strength=0.7, noise_level=0.1, seed=None):
    """
    Generate synthetic time series with causal relationships.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of time points
    causal_strength : float, default=0.7
        Strength of causal relationships
    noise_level : float, default=0.1
        Amount of noise to add
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X1, X2, Y : numpy.ndarray
        Time series with causal relationships:
        X1 → Y (with lag 1)
        X2 → Y (with lag 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate independent sources
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    
    # Generate target with causal influence from X1 (lag 1) and X2 (lag 2)
    Y = np.zeros(n_samples)
    
    for t in range(n_samples):
        if t >= 1:
            # Influence from X1 with lag 1
            Y[t] += causal_strength * X1[t-1]
        if t >= 2:
            # Influence from X2 with lag 2
            Y[t] += causal_strength * X2[t-2]
        
        # Add noise
        Y[t] += noise_level * np.random.randn()
    
    return X1, X2, Y

def compare_mi_di(X1, X2, Y, max_lag=5, bins=10, save_path=None):
    """
    Compare mutual information and directed information approaches.
    
    Parameters:
    -----------
    X1, X2, Y : numpy.ndarray
        Time series data
    max_lag : int, default=5
        Maximum time lag to consider
    bins : int, default=10
        Number of bins for discretization
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    comparison : dict
        Dictionary with comparison results
    """
    # Calculate mutual information (static)
    mi_x1y = mutual_info_score(
        np.digitize(X1, np.linspace(min(X1), max(X1), bins+1)),
        np.digitize(Y, np.linspace(min(Y), max(Y), bins+1))
    ) / np.log(2)
    
    mi_x2y = mutual_info_score(
        np.digitize(X2, np.linspace(min(X2), max(X2), bins+1)),
        np.digitize(Y, np.linspace(min(Y), max(Y), bins+1))
    ) / np.log(2)
    
    # Calculate directed information for different lags
    di_x1y = np.zeros(max_lag+1)
    di_x2y = np.zeros(max_lag+1)
    
    for lag in range(1, max_lag+1):
        di_x1y[lag] = estimate_transfer_entropy(X1, Y, lag, bins)
        di_x2y[lag] = estimate_transfer_entropy(X2, Y, lag, bins)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # MI comparison
    plt.subplot(1, 2, 1)
    plt.bar(['X1→Y', 'X2→Y'], [mi_x1y, mi_x2y], color=['blue', 'orange'])
    plt.title('Mutual Information (Static)', fontsize=24)
    plt.ylabel('Information (bits)', fontsize=20)
    
    # DI comparison
    plt.subplot(1, 2, 2)
    plt.plot(range(1, max_lag+1), di_x1y[1:], 'bo-', label='X1→Y')
    plt.plot(range(1, max_lag+1), di_x2y[1:], 'ro-', label='X2→Y')
    plt.title('Directed Information (Transfer Entropy)', fontsize=24)
    plt.xlabel('Time Lag', fontsize=20)
    plt.ylabel('Information (bits)', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    # Create comparison dictionary
    comparison = {
        'mi_x1y': mi_x1y,
        'mi_x2y': mi_x2y,
        'di_x1y': di_x1y[1:],
        'di_x2y': di_x2y[1:],
    }
    
    return comparison

# Example usage
if __name__ == "__main__":
    print("Generating synthetic time series data...")
    X1, X2, Y = generate_causal_time_series(n_samples=1000, causal_strength=0.7, noise_level=0.2, seed=42)
    
    print("Comparing mutual information and directed information...")
    # Create results directory if it doesn't exist
    if not os.path.exists('../results'):
        os.makedirs('../results')
        
    compare_mi_di(X1, X2, Y, max_lag=5, bins=8, save_path='../results/mi_vs_di_comparison.png')
    
    print("Performing multi-lag PID analysis...")
    results = multi_lag_analysis(X1, X2, Y, max_lag=5, bins=8)
    
    print("Plotting PID results...")
    plot_multi_lag_results(results, save_path='../results/temporal_pid_results.png')
    
    print("Analysis complete. Results saved to '../results/' directory.") 