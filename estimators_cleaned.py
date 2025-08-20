import numpy as np
from typing import Union, Tuple, List, Optional, Callable, Dict, Any
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
    
def get_laplacian_matrices(adjacency_matrices: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute Laplacian matrices from adjacency matrices.
    """
    D = [np.diag((1e-10 + A.sum(axis=1))**-0.5) for A in adjacency_matrices]
    return [D @ A @ D for A,D in zip(adjacency_matrices, D)]


def unfolded_adjacency_spectral_embedding(
        adjacency_matrices: List[np.ndarray],
        d: Optional[int] = None,
        threshold: Optional[float] = 0.2
    ) -> List[np.ndarray]:
    """
    Unfolded Adjacency Spectral Embedding (UASE) for dynamic networks.
    
    Parameters:
    -----------
    adjacency_matrices : List[np.ndarray]
        List of T adjacency matrices, each n x n
    d : int
        Fixed embedding dimension. 

    TODO:  Remove threshold (original suggestion from hierarchical paper)
        
    Returns:
    --------
    Y_hat_list : List[np.ndarray]
        List of T embedding matrices, each n x d
    """

    # Check inputs
    if not adjacency_matrices:
        raise ValueError("adjacency_matrices cannot be empty")
    
    # Check all matrices have same shape and are square
    n = adjacency_matrices[0].shape[0]
    T = len(adjacency_matrices)
    
    for i, A in enumerate(adjacency_matrices):
        if A.shape != (n, n):
            raise ValueError(f"All matrices must be {n}x{n}, but matrix {i} has shape {A.shape}")
    
    # Stack matrices horizontally
    A_unfolded = np.hstack(adjacency_matrices)  # n x (n*T)
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(A_unfolded, full_matrices=False)
    V = Vt.T
    
    # Determine dimension
    if d is not None:
        if d > n:
            raise ValueError(f"d ({d}) cannot be larger than n = {n}")
        d_use = d
    elif threshold is not None:
        if threshold <= 0 or threshold > 1:
            raise ValueError("threshold must be in (0, 1]")
        max_singular_value = s[0]
        if max_singular_value > 0:
            # Find singular values that meet threshold
            mask = s >= threshold * max_singular_value
            d_use = np.sum(mask)
            if d_use == 0:
                d_use = 1  # Always keep at least one dimension
        else:
            # All singular values are zero
            d_use = 1
    else:
        raise ValueError("Either d or threshold must be specified")
    
    # Extract first d components
    V_d = V[:, :d_use]
    s_d = s[:d_use]
    
    # Compute embedding: Y_hat = V_d * Sigma_d^(1/2)
    Sigma_sqrt = np.diag(np.sqrt(s_d))
    Y_hat_unfolded = V_d @ Sigma_sqrt  # (n*T) x d
    
    # Split back into time slices
    Y_hat_list = []
    for t in range(T):
        start_idx = t * n
        end_idx = (t + 1) * n
        Y_hat_t = Y_hat_unfolded[start_idx:end_idx, :]
        Y_hat_list.append(Y_hat_t)
    
    return Y_hat_list


def cartesian_to_spherical(X: np.ndarray) -> np.ndarray:
    """
    Normalize Cartesian coordinate embeddings to spherical coordinates.
    
    Following equaton (eq:polar_coords) from the paper Passino et al (2013)
    θ_j = 
        arccos(x_2/||x_{:2}||)              if j=1, x_1 >= 0
        2 pi - arccos(x_2/||x_{:2}||)       if j=1, x_1 < 0
        2 pi * arccos(x_{j+1}/||x_{:j+1}||) if j >= 2
    
    NB, paper uses 1-based indexing, we convert to 0-based.
    
    Parameters:
    -----------
    X : np.ndarray
        n x d matrix of Cartesian coordinates
        
    Returns:
    --------
    Theta : np.ndarray
        n x (d-1) matrix of spherical coordinates
    """
    n, d = X.shape
    if d < 2:
        raise ValueError("Need at least 2 dimensions for spherical coordinates")
    
    Theta = np.zeros((n, d-1))
    
    for i in range(n):
        x = X[i]  # Process row i
        
        # First angle (j=1 in paper, index 0 for us)
        # \theta_0 = arccos(x_2/||x_{:2}||) with adjusment for sign of x_1
        norm_2d = np.linalg.norm(x[:2])
        if norm_2d > 1e-10:
            cos_angle = x[1] / norm_2d  # x_2 in paper notation
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            if x[0] < 0:  # x_1 in paper notation
                angle = 2 * np.pi - angle
                
            Theta[i, 0] = angle
        else:
            Theta[i, 0] = 0
        
        # Remaining angles (j >= 2 in paper, indices 1+ for us)
        for j in range(1, d-1):
            # Paper formula: \theta_j = 2 * arccos(x_{j+1}/||x_{:j+1}||)
            # In 0-based: \theta[j] = 2 * arccos(x[j+1]/||x[:j+2]||)
            norm_partial = np.linalg.norm(x[:j+2])
            
            if norm_partial > 1e-10:
                cos_angle = x[j+1] / norm_partial
                Theta[i, j] = 2 * np.arccos(np.clip(cos_angle, -1, 1))
            else:
                Theta[i, j] = 0
    
    return Theta


def estimate_d_regularized_aic(
        adjacency_matrices: List[np.ndarray],
        d_max: Optional[int] = 20,
        d_min: Optional[int] = 2,
        regularization_strength: float = 3.5, 
        verbose: bool = False
    ) -> Tuple[int, dict]:
    """
    Estimate embedding dimension using regularized AIC.
    
    This method uses AIC with an adjustable regularization parameter that
    controls the penalty for model complexity. Higher regularization values
    lead to simpler models (smaller d).

    The default value of 3.5 was found heuristically to work well.  
    
    Parameters
    ----------
    adjacency_matrices : List[np.ndarray]
        List of T adjacency matrices, each n x n
    d_max : int, optional 
        Maximum dimension to consider.  Default 20 is quite large actually.  
    d_min : int, optional 
        Minimum dimension 
    regularization_strength : float, optional 
        Multiplier for the penalty term
    verbose : bool
        Print progress information
        
    Returns
    -------
    d_est : int
        Estimated embedding dimension
    info : dict
        Dictionary
        - 'aic_values': Regularized AIC values for each dimesion
        - 'log_likelihoods': Log-likelihood values
        - 'singular_values': Singular values of unfolded adjacency
        - 'd_range': Range of dimensions tested
        - 'regularization_strength': Value used
    """
    n = adjacency_matrices[0].shape[0]
    T = len(adjacency_matrices)
    
    # adj -> laplacian 
    laplacian_matrices = get_laplacian_matrices(adjacency_matrices)
    L = np.hstack(laplacian_matrices)
    
    # Compute SVD once
    U, s, Vt = np.linalg.svd(L, full_matrices=False)
    d_max = min(d_max, len(s))
    d_range = range(d_min, d_max + 1)
    
    if verbose:
        print(f"Estimating embedding dimension using regularized AIC")
        print(f"Regularization strength: {regularization_strength}")
        print(f"Testing d = {d_min} to {d_max}")
    
    log_likelihoods = []
    aic_values = []
    
    # Mask for upper triangle to aviod double counting
    mask = np.triu(np.ones((n, n)), k=1)
    mask_full = np.tile(mask, (1, T))
    
    for d in d_range:
        # Rank-d approximation
        Ld = U[:, :d] @ np.diag(s[:d]) @ Vt[:d, :]
        
        # Approximate prob matrix 
        Pd = np.clip(Ld, 1e-10, 1 - 1e-10)
        
        # Log-likelihood (Bernoulli model)
        log_lik = np.sum(
            mask_full * (L * np.log(Pd) + (1 - L) * np.log(1 - Pd))
        )
        log_likelihoods.append(log_lik)
        
        # parameters:  nd parameters in Ud, nTd parameters in Vd
        # and orthogonality in d dimensions  
        n_params = d * (n + n*T - d)
        
        # Regularized AIC = 2 * regularization_strength * k - 2 * ln(L) 
        aic = 2 * regularization_strength * n_params - 2 * log_lik
        aic_values.append(aic)
    
    # Select d with minimum regularized AIC
    d_est_idx = np.argmin(aic_values)
    d_est = list(d_range)[d_est_idx]
    
    if verbose:
        print(f"\nSelected d = {d_est}")
        print(f"Regularized AIC at d_est: {aic_values[d_est_idx]:.1f}")
        print(f"Log-likelihood at d_est: {log_likelihoods[d_est_idx]:.1f}")
    
    info = {# TODO:  Shorten this, we will only use d_est_idx 
        'aic_values': np.array(aic_values),
        'log_likelihoods': np.array(log_likelihoods),
        'singular_values': s[:d_max],
        'd_range': list(d_range),
        'd_est_idx': d_est_idx,
        'regularization_strength': regularization_strength
    }
    
    return d_est, info


#TODO:  Remove 
def gmm_clustering(
        X: np.ndarray, 
        n_clusters: int, 
        covariance_type: str = 'full', # TODO: Remove, always use full
        n_init: int = 10, # TODO: Remove, always use 10
        max_iter: int = 300 # TODO: Remove, always use 300
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian Mixture Model clustering for spectral embeddings.
    
    Parameters:
    -----------
    X : np.ndarray
        n x d latent embedding 
    n_clusters : int
        Number of clusters
    covariance_type : str
        Type of covariance parameters to use ('full', 'tied', 'diag', 'spherical')
    n_init : int
        Number of inital clusters 
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    labels : np.ndarray
        Cluster assignments for each point
    centers : np.ndarray
        Cluster centers (means of Gaussians)
    """
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter
    )
    
    labels = gmm.fit_predict(X)
    centers = gmm.means_
    
    return labels, centers


def dynamic_dcsbm_clustering(
        adjacency_matrices: List[np.ndarray],
        R: int,
        d: Optional[int] = None,
        threshold: float = 0.1,  #TODO:  delete 
        max_dim: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:

    
    """
    Estimate communities in a dynamic DCSBM using UASE + spherical coordinates + GMM.
    
    Parameters:
    -----------
    adjacency_matrices : List[np.ndarray]
        List of T adjacency matrices, each n x n
    R : int
        Number of communities to find
    max_dim : int, optional
        Maximum number of spherical dimensions to use for clustering.
        If None, uses all available dimensions.
        
    Returns
    -------
    labels : np.ndarray
        Community labels of shape (T, n)
    info : dict
        Dictionary containing:
        - 'd_cartesian': Cartesian embedding dimension
        - 'd_spherical': Spherical dimension
        - 'd_used': Actual dimensions used for clustering
        - 'Y_list': Cartesian UASE embeddings
        - 'Y_spherical_list': Spherical embeddings
    """
    n = adjacency_matrices[0].shape[0]
    T = len(adjacency_matrices)
    
    # Step 1: UASE embedding
    Y_list = unfolded_adjacency_spectral_embedding(adjacency_matrices, d = d)
    d_cartesian = Y_list[0].shape[1]
    
    # Handle low-dimensional case
    if d_cartesian < 2:
        # Can't do spherical transformation
        Y_stacked = np.vstack(Y_list)
        labels_stacked, _ = gmm_clustering(Y_stacked, n_clusters=R)
        labels = labels_stacked.reshape(T, n)
        
        return labels, {
            'd_cartesian': d_cartesian,
            'd_spherical': 0,
            'd_used': d_cartesian,
            'Y_list': Y_list,
            'Y_spherical_list': None
        }
    
    # Step 2: Convert to spherical coordinates
    Y_spherical_list = [cartesian_to_spherical(Y) for Y in Y_list]
    d_spherical = Y_spherical_list[0].shape[1]
    
    # Step 3: Determine dimensions to use
    if max_dim is not None:
        d_used = min(max_dim, d_spherical)
        Y_clustering = [Y[:, :d_used] for Y in Y_spherical_list]
    else:
        d_used = d_spherical
        Y_clustering = Y_spherical_list
    
    # Step 4: Stack all embeddings across time
    Y_stacked = np.vstack(Y_clustering)
    
    # Step 5: Cluster using GMM
    labels_stacked, centers = gmm_clustering(Y_stacked, n_clusters=R)
    
    # Step 6: Reshape labels back to (T, n)
    labels = labels_stacked.reshape(T, n)
    
    info = {# TODO:  remove, I think we will only use labels from now 
        'd_cartesian': d_cartesian,
        'd_spherical': d_spherical,
        'd_used': d_used,
        'Y_list': Y_list,
        'Y_spherical_list': Y_spherical_list
    }
    
    return labels, info


def estimate_R_silhouette(
        Y_spherical_list: List[np.ndarray],
        R_max: Optional[int] = 10,
        R_min: int = 2,
        n_init: int = 50,
        verbose: bool = False
    ) -> Tuple[int, dict]:
    """
    Estimate number of communities using silhouette score and KMeans on spherical embeddings.
    
    Parameters
    ----------
    Y_spherical_list : List[np.ndarray]
        List of spherical embeddings, one per time point
    R_max : int, optional
        Maximum number of communities to consider. 
    R_min : int, optional 
        Minimum number of communities.  
    n_init : int  optional
        Number of k-means initializations.  
    verbose : bool
        Print progress information
        
    Returns
    ------
    R_est : int
        Estimated number of communities
    info : dict
        Dictionary with 
        - 'silhoette_scores': Silhouette scores for each R
        - 'stability_scores': Temporal stability scores (if T > 1)
        - 'combined_scores': Combined scores used for selection
        - 'R_range': Range of R values tested
        - 'best_score': Best combined score achieved
        - 'n_warnings': Number of k-means convergence warnings
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    # stack and find dimensions 
    Y_stacked = np.vstack(Y_spherical_list)
    n = Y_spherical_list[0].shape[0]
    T = len(Y_spherical_list)
      
    if verbose:
        print(f"Estimating number of communities using silhouette score")
        print(f"Data: {n} nodes × {T} time points")
        print(f"Testing R = {R_min} to {R_max}")
        if T > 1 and stability_weight > 0:
            print(f"Using temporal stability weight: {stability_weight}")
    
    # Test different numbers of clusters
    R_range = list(range(R_min, R_max + 1))
    silhouette_scores = []
    stability_scores = []
    combined_scores = []
    n_warnings = 0
    cluster_results = []
    
    for R in R_range:
        # Cluster all time points jointly
        kmeans = KMeans(n_clusters=R, n_init=n_init)
        labels_stacked = kmeans.fit_predict(Y_stacked)
        
        # Check if k-means found fewer clusters
        n_found = len(np.unique(labels_stacked))
        if n_found < R:
            n_warnings += 1
        
        # Compute silhouette score
        if n_found >= 2:
            sil_score = silhouette_score(Y_stacked, labels_stacked)
        else:
            sil_score = -1
            
        silhouette_scores.append(sil_score)
        
        combined_scores.append(sil_score)

        # Store results
        cluster_results.append({
            'R': R,
            'n_found': n_found,
            'labels': labels_stacked,
            'silhouette': sil_score,
            # 'stability': stability,
            # 'combined': combined
        })
        
        if verbose:
            warning_str = f" (found only {n_found})" if n_found < R else ""
            print(f"  R={R}: silhouette={sil_score:.3f}, "
                  f"stability={stability:.3f}, combined={combined:.3f}{warning_str}")


    info = {
        'silhouette_scores': np.array(silhouette_scores),
        'stability_scores': np.array(stability_scores),
        'combined_scores': np.array(combined_scores),
        'R_range': R_range,
        'best_score': -np.inf,
        'n_warnings': n_warnings,
        'cluster_results': cluster_results
    }

    # Select R with maximum combined score
    if not combined_scores:
        return R_range[0], info
    
    best_idx = np.argmax(combined_scores)
    R_est = R_range[best_idx]
    info['best_score'] = combined_scores[best_idx]
    
    if verbose:
        print(f"\nSelected R = {R_est} (combined score = {info['best_score']:.3f})")
    

    
    return R_est, info


def compute_temporal_stability(Y_list, d, verbose=False):
    """
    Compute temporal stability for a given time series of embeddings.
    High stability = low temporal variance relative to spatial spread.
    """
    n = Y_list[0].shape[0]
    T = len(Y_list)
    stability_scores = []

    for dim in range(d):

        # need to re-dim for extracting one dimension 
        dim_values = np.array([Y_list[t][:, dim] for t in range(T)])  # Shape: (T, n)
        
        # calculate average spatial vs temporal variance 
        spatial_vars = [np.var(dim_values[t, :]) for t in range(T)]
        avg_spatial_var = np.mean(spatial_vars)
        
        node_temporal_vars = np.var(dim_values, axis=0)  
        avg_temporal_var = np.mean(node_temporal_vars)
        
        # temporal; stability = spatial var (signal) / temproral var (noise)
        stability = avg_spatial_var / (avg_temporal_var + 1e-10)
        stability_scores.append(stability)
        if verbose:
            print(f"Dim {dim}: spatial var = {avg_spatial_var:.6f}, "
                f"temporal var = {avg_temporal_var:.6f}, ratio = {stability:.3f}")
    
    return np.array(stability_scores)


def find_elbow_cutoff(sorted_scores, significance_level=0.01, verbose=True):    
    """
    Find elbow in sorted stability scores using perpendicular distance method.

    We use log scale because some scores vary from 30 to 0.2.  We assume the scores 
    to correspond to even-distanced inputs (eg dimensions).  

    Parameters:
    ------------ 
    sorted_scores: np.ndarray
        Sorted stability scores
    verbose: bool

    Returns:
    --------
    best_idx: int   
        Index of the elbow

    TODO:  Remove significane level
    """
    n = len(sorted_scores)
    
    log_scores = np.log(sorted_scores + 1e-10)
    
    # get into Euclidean space 
    x = np.arange(n)
    y = log_scores
    
    # Line from first to last point
    x1, y1 = 0, log_scores[0]
    x2, y2 = n-1, log_scores[-1]
    
    # Calculate perpendicular distance from each point to the line
    distances = []
    
    # Line equation: ax + by + c = 0
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1
    
    # Distance from point (x0, y0) to line ax + by + c = 0:
    # d = |ax0 + by0 + c| / sqrt(a^2 + b^2)
    denominator = np.sqrt(a**2 + b**2)
    
    for i in range(1, n-1):  # Skip first and last points
        distance = abs(a * x[i] + b * y[i] + c) / denominator
        distances.append(distance)
    
    # Find point with maximum distance
    if distances:
        best_idx = np.argmax(distances) + 1  # +1 because we started from index 1
    else:
        best_idx = 1
    
    if verbose:
        print(f"\nElbow detection:")
        print(f"  Elbow at dimension: {best_idx}")
        print(f"  Max perpendicular distance: {distances[best_idx-1]:.4f}")
        print(f"  Score at elbow: {sorted_scores[best_idx]:.4f}")
        print(f"  Score drop: {sorted_scores[0]/sorted_scores[best_idx]:.2f}x")
        
    return best_idx


def estimate_n_clusters(Y_stable, max_k=10, verbose=True):
    """
    Estimate optimal number of clusters using silhouette score over KMeans.

    Parameters:
    -----------
    Y_stable: np.ndarray
        Stable embedding of shape (n, d)
    max_k: int, optional
        Maximum number of clusters to consider
    verbose: bool
        Print info
    """

    if verbose:
        print("\nEstimating number of clusters...")
    
    scores = []
    K_range = range(2, min(max_k + 1, Y_stable.shape[0] // 10))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(Y_stable)
        score = silhouette_score(Y_stable, labels)
        scores.append(score)
        
        if verbose:
            print(f"  k={k}: silhouette = {score:.3f}")
    
    # Find best k
    best_k = K_range[np.argmax(scores)]
    
    if verbose:
        print(f"  Selected k={best_k}")
    
    return best_k


def detect_super_communities_simple(adjacency_matrices, d=20, significance_level=0.01, 
                                   n_super_communities=None, verbose=False):
    """
    Simplified pipeline for super-community detection using temporal stability.
    
    Parameters
    ----------
    adjacency_matrices : List[np.ndarray]
        List of T adjacency matrices, each n x n
    d : int
        Initial UASE embedding dimension
    significance_level : float
        Significance level for dimension cutoff (default 0.01)
    n_super_communities : int, optional
        Number of super-communities. If None, estimate from data.
    verbose : bool
        Print progress information
        
    Returns
    -------
    labels : np.ndarray
        Super-community assignments for each node
    analytics : dict
        Dictionary with analytics data
    """
    n = adjacency_matrices[0].shape[0]
    

    # Step 0:  normalize adjacency matrices 
    laplacian_matrices = get_laplacian_matrices(adjacency_matrices)

    # Step 1: SVD 
    U, S, Vt = np.linalg.svd(laplacian_matrices[0])
    V = Vt.T
    singular_values = S
    dim_order = np.argsort(singular_values)[::-1]
    sorted_scores = singular_values[dim_order]
    n_stable = find_elbow_cutoff(sorted_scores, significance_level, verbose=False)
    stable_dims = dim_order[:n_stable]

    Y = V[:, stable_dims] @ np.diag(np.sqrt(singular_values[stable_dims]))

    # back to list 
    Y_list = [Y[:, t*n:(t+1)*n] for t in range(len(adjacency_matrices))]
    
    if verbose:
        print(f"Selected {n_stable} stable dimensions: {stable_dims}")
    
    
    # Step 5: Estimate number of clusters if needed
    if n_super_communities is None:
        n_super_communities = estimate_n_clusters(Y, max_k=10, verbose=verbose)
    
    # Step 6: Cluster
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    kmeans = KMeans(n_clusters=n_super_communities, n_init=50)
    labels_stacked = kmeans.fit_predict(Y)
    labels = labels_stacked[:n]  # Take first time point labels
    
    # Compute quality metric
    sil_score = silhouette_score(Y, labels_stacked) if n_super_communities > 1 else 0
    
    if verbose:
        print(f"\nSuper-community sizes:")
        for k in range(n_super_communities):
            size = np.sum(labels == k)
            print(f"  Super-community {k}: {size} nodes ({size/n*100:.1f}%)")
        print(f"Silhouette score: {sil_score:.3f}")
    


    # Package analytics
    analytics = {
        'Y_list': Y_list,
        'Y_stable_list': Y_list,
        'stable_dims': stable_dims,
        'stability_scores': S,
        'n_stable': n_stable,
        'kmeans_model': kmeans,
        'silhouette_score': sil_score,
        'dim_order': dim_order,
        'sorted_scores': sorted_scores
    }
    
    return labels, analytics


def detect_super_communities_stable(adjacency_matrices, D_init=20, significance_level=0.01, 
                                   n_super_communities=None, verbose=False):
    """
    Super-community detection using temporal stability.
    
    Parameters
    ----------
    adjacency_matrices : List[np.ndarray]
        List of T adjacency matrices, each n x n
    D_init : int
        Initial UASE embedding dimension
    significance_level : float
        Significance level for dimension cutoff (default 0.01)
    TODO:  This is not used downstream, remove.  
    n_super_communities : int, optional
        Number of super-communities. If None, estimate from data.
    verbose : bool
        Print progress information
        
    Returns
    -------
    labels : np.ndarray
        Super-community assignments for each node
    analytics : dict
        Dictionary with analytics data
    """
    n = adjacency_matrices[0].shape[0]
    
    laplacian_matrices = get_laplacian_matrices(adjacency_matrices)

    # Step 1: UASE embeddings of large dimension 
    if verbose:
        print("Computing UASE embeddings...")
    Y_list = unfolded_adjacency_spectral_embedding(laplacian_matrices, d=D_init)
    
    # Step 2: Compute temporal stability
    if verbose:
        print("Computing temporal stability scores...")
    # Pass the dimension explicitly
    stability_scores = compute_temporal_stability(Y_list, d=D_init, verbose=verbose)
    
    # Step 3: Find stable dimensions
    dim_order = np.argsort(stability_scores)[::-1]
    sorted_scores = stability_scores[dim_order]
    n_stable = find_elbow_cutoff(sorted_scores, significance_level, verbose=False)
    stable_dims = dim_order[:n_stable]
    
    if verbose:
        print(f"Selected {n_stable} stable dimensions: {stable_dims}")
    
    # Step 4: Project to stable subspace
    Y_stable_list = [Y[:, stable_dims] for Y in Y_list]
    Y_stable_stacked = np.vstack(Y_stable_list)
    
    # Step 5: Estimate number of clusters if needed
    if n_super_communities is None:
        n_super_communities = estimate_n_clusters(Y_stable_stacked, max_k=10, verbose=verbose)
    
    # Step 6: Cluster
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    kmeans = KMeans(n_clusters=n_super_communities, n_init=50)
    labels_stacked = kmeans.fit_predict(Y_stable_stacked)
    labels = labels_stacked[:n]  # Take first time point labels
    
    # Compute quality metric
    sil_score = silhouette_score(Y_stable_stacked, labels_stacked) if n_super_communities > 1 else 0
    
    if verbose:
        print(f"\nSuper-community sizes:")
        for k in range(n_super_communities):
            size = np.sum(labels == k)
            print(f"  Super-community {k}: {size} nodes ({size/n*100:.1f}%)")
        print(f"Silhouette score: {sil_score:.3f}")
    
    # Package analytics
    analytics = {
        'Y_list': Y_list,
        'Y_stable_list': Y_stable_list,
        'stable_dims': stable_dims,
        'stability_scores': stability_scores,
        'n_stable': n_stable,
        'kmeans_model': kmeans,
        'silhouette_score': sil_score,
        'dim_order': dim_order,
        'sorted_scores': sorted_scores
    }
    
    return labels, analytics


def dynamic_hierarchical_dcsbm_detection_simple(
    adjacency_matrices: List[np.ndarray],
    D: Optional[int] = None,
    R_super: Optional[int] = None,
    d_subgraph: Optional[int] = None,
    R_subgraph: Optional[List[int]] = None,
    significance_level: float = 0.01,
    d_threshold: float = 0.1,
    regularization_strength: float = 3.5,
    stability_weight: float = 0.0,
    verbose: bool = False,
    plot_diagnostics: bool = False,
) -> Dict:
    """
    Variant of the main algorithm, using not temporally stable but standard UASE embeddings.  
        
    This algorithm combines:
    1. Super-community detection from UASE embeddings interpreted as paths in R^d per node.  
    2. Sub-community detection within each super-community using dynamic DC-SBM methods
    
    Parameters
    ----------
    adjacency_matrices : List[np.ndarray]
        List of T adjacency matrices, each n x n
    D : int, optional
        Initial UASE dimension. If None, use 20.
    R_super : int, optional
        Number of super-communities. If None, estimated automatically.
    d_subgraph : int, optional
        Subgraph embedding dimension. If None, estimated per subgraph.
    R_subgraph : List[int], optional
        Number of communities per subgraph. If None, estimated per subgraph.
    TODO: Remove d_threshold, stability_weight
    regularization_strength : float
        Regularization for AIC-based dimension estimation 
    verbose : bool
        Print detailed progress
    plot_diagnostics : bool
        Show diagnostic plots
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'super_communities': Static super-community assignments dim n
        - 'communities': Dynamic community assignments dim Txn
        - 'hierarchical_labels': List of (super, sub) tuples per time dim Txn
        - 'R_super_est': Estimated number of super-communities
        - 'd_est_per_subgraph': Dict of embedding dimensions per subgraph (can remove) 
        - 'R_est_per_subgraph': Dict of community counts per subgraph (can remove) 
        - 'super_community_analytics': Analytics from super-community detection (can remove) 
        - 'subgraph_results': Detailed results per subgraph (can remove) 
    """
    n = adjacency_matrices[0].shape[0]
    T = len(adjacency_matrices)
    
    if verbose:
        print("="*70)
        print("DYNAMIC HIERARCHICAL DC-SBM COMMUNITY DETECTION")
        print("(Using Temporal Stability Method)")
        print("="*70)
        print(f"Data: {n} nodes, {T} time points")
        edge_density = np.mean([np.sum(A)/(n*(n-1)) for A in adjacency_matrices])
        print(f"Edge density: {edge_density:.4f}")
    
    # =========================================================================
    # STEPS 1-4: SUPER-COMMUNITY DETECTION 
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("SUPER-COMMUNITY DETECTION")
        print("="*70)
    
    # Use default D if not provided
    if D is None:
        D = 20
        if verbose:
            print(f"Using default initial dimension D = {D}")
    
    # Detect super-communities 
    super_community_labels, super_analytics = detect_super_communities_simple(
        adjacency_matrices,
        d=D,
        significance_level=significance_level,
        n_super_communities=R_super,
        verbose=verbose
    )
    
    R_super_est = len(np.unique(super_community_labels))
    
    if plot_diagnostics:
        # Plot stability scores and cutoff
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Stability scores
        scores = super_analytics['sorted_scores']
        n_stable = super_analytics['n_stable']
        ax1.plot(range(len(scores)), scores, 'o-', markersize=8)
        ax1.axvline(x=n_stable-0.5, color='red', linestyle='--', 
                   label=f'Cutoff ({n_stable} dims)')
        ax1.set_xlabel('Dimension Rank')
        ax1.set_ylabel('Stability Score')
        ax1.set_title('Temporal Stability Scores')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: First two stable dimensions
        Y_stable_0 = super_analytics['Y_stable_list'][0]
        if Y_stable_0.shape[1] >= 2:
            for k in range(R_super_est):
                mask = super_community_labels == k
                ax2.scatter(Y_stable_0[mask, 0], Y_stable_0[mask, 1], 
                          label=f'SC {k}', alpha=0.6, s=20)
            ax2.set_xlabel('Stable Dimension 1')
            ax2.set_ylabel('Stable Dimension 2')
            ax2.set_title('Super-Communities in Stable Subspace')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # =========================================================================
    # STEP 5: PROCESS EACH SUPER-COMMUNITY SUBGRAPH
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("SUB-COMMUNITY DETECTION WITHIN SUPER-COMMUNITIES")
        print("="*70)
    
    # Initialize storage for results
    d_est_per_subgraph = {}
    R_est_per_subgraph = {}
    subgraph_results = {}
    
    # Create index mapper
    from dyn_hdcsbm import SubgraphIndexMapper
    mapper = SubgraphIndexMapper(super_community_labels)
    
    for sg_id in range(R_super_est):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing Super-Community {sg_id}")
        
        # Get subgraph nodes
        sg_nodes = mapper.get_subgraph_nodes(sg_id)
        n_sg = len(sg_nodes)
        
        if n_sg < 10:  # Skip if too small
            if verbose:
                print(f"  Skipping: only {n_sg} nodes")
            d_est_per_subgraph[sg_id] = 1
            R_est_per_subgraph[sg_id] = 1
            subgraph_results[sg_id] = {
                'labels': np.zeros((T, n_sg), dtype=int),
                'message': 'Subgraph too small'
            }
            continue
        
        # Extract subgraph adjacency matrices
        A_subgraph_list = []
        for t in range(T):
            A_t = adjacency_matrices[t]
            A_sg = A_t[np.ix_(sg_nodes, sg_nodes)]
            A_subgraph_list.append(A_sg)
        
        # Step 5.1: Estimate embedding dimension for subgraph
        if d_subgraph is None:
            if verbose:
                print(f"\n  Step 5.{sg_id}.1: Estimating embedding dimension")
            
            d_sg_est, d_info = estimate_d_regularized_aic(
                A_subgraph_list,
                d_max=min(20, n_sg//2),
                regularization_strength=regularization_strength,
                verbose=False
            )
            d_est_per_subgraph[sg_id] = d_sg_est
            
            if verbose:
                print(f"    Estimated d = {d_sg_est}")
        else:
            d_sg_est = d_subgraph
            d_est_per_subgraph[sg_id] = d_sg_est
        
        # Step 5.2: Estimate number of communities in subgraph
        if R_subgraph is None:
            if verbose:
                print(f"\n  Step 5.{sg_id}.2: Estimating number of communities")
            
            # Get UASE embeddings for subgraph
            Y_sg_list = unfolded_adjacency_spectral_embedding(A_subgraph_list, d=d_sg_est)
            
            # Convert to spherical if possible
            if d_sg_est >= 2:
                Y_sg_spherical = [cartesian_to_spherical(Y) for Y in Y_sg_list]
            else:
                Y_sg_spherical = Y_sg_list
            
            R_sg_est, R_info = estimate_R_silhouette(
                Y_sg_spherical,
                R_max=min(10, n_sg//10),
                # stability_weight=stability_weight,
                verbose=False
            )
            R_est_per_subgraph[sg_id] = R_sg_est
            
            if verbose:
                print(f"    Estimated R = {R_sg_est}")
        else:
            R_sg_est = R_subgraph[sg_id] if isinstance(R_subgraph, list) else R_subgraph
            R_est_per_subgraph[sg_id] = R_sg_est
        
        # Step 5.3: Perform dynamic community detection
        if verbose:
            print(f"\n  Step 5.{sg_id}.3: Dynamic community detection")
        
        sg_labels, sg_info = dynamic_dcsbm_clustering(
            A_subgraph_list,
            d=d_sg_est,
            R=R_sg_est
        )
        
        subgraph_results[sg_id] = {
            'labels': sg_labels,  # Shape (T, n_sg)
            'info': sg_info,
            'd_est': d_sg_est,
            'R_est': R_sg_est,
            'nodes': sg_nodes
        }
        
        if verbose:
            # Print community sizes at t=0
            labels_t0 = sg_labels[0]
            print(f"    Community sizes at t=0:")
            for comm in range(R_sg_est):
                size = np.sum(labels_t0 == comm)
                print(f"      Community {comm}: {size} nodes")
    
    # =========================================================================
    # STEP 6: COMBINE RESULTS
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("COMBINING RESULTS")
        print("-"*70)
    
    # Create full community labels array
    community_labels = np.zeros((T, n), dtype=int)
    hierarchical_labels = []
    
    for t in range(T):
        labels_t = []
        for i in range(n):
            sg_id = super_community_labels[i]
            
            if sg_id in subgraph_results and 'labels' in subgraph_results[sg_id]:
                # Get local index within subgraph
                sg_nodes = subgraph_results[sg_id]['nodes']
                local_idx = np.where(sg_nodes == i)[0][0]
                
                # Get community within subgraph
                comm_id = subgraph_results[sg_id]['labels'][t, local_idx]
                
                # Create global community ID
                offset = sum(R_est_per_subgraph[j] for j in range(sg_id))
                global_comm_id = offset + comm_id
                
                community_labels[t, i] = global_comm_id
                labels_t.append((sg_id, comm_id))
            else:
                # Handle edge cases
                community_labels[t, i] = -1
                labels_t.append((sg_id, -1))
        
        hierarchical_labels.append(labels_t)
    
    # =========================================================================
    # COMPILE FINAL RESULTS
    # =========================================================================
    results = {
        'super_communities': super_community_labels,
        'communities': community_labels,
        'hierarchical_labels': hierarchical_labels,
        'R_super_est': R_super_est,
        'd_est_per_subgraph': d_est_per_subgraph,
        'R_est_per_subgraph': R_est_per_subgraph,
        'super_community_analytics': super_analytics,
        'subgraph_results': subgraph_results
    }
    
    if verbose:
        print("\n" + "="*70)
        print("ALGORITHM COMPLETE")
        print("="*70)
        print(f"Super-communities: {R_super_est}")
        print(f"Total communities: {sum(R_est_per_subgraph.values())}")
        print(f"Hierarchical structure:")
        for sg_id in range(R_super_est):
            print(f"  Super-community {sg_id}: {R_est_per_subgraph.get(sg_id, 0)} communities")
    
    return results


def dynamic_hierarchical_dcsbm_detection_stable(
    adjacency_matrices: List[np.ndarray],
    D: Optional[int] = None,
    R_super: Optional[int] = None,
    d_subgraph: Optional[int] = None,
    R_subgraph: Optional[List[int]] = None,
    significance_level: float = 0.01, 
    d_threshold: float = 0.1, #TODO:  delete 
    regularization_strength: float = 3.5, 
    stability_weight: float = 0.0, #TODO:  delete 
    verbose: bool = False,
    plot_diagnostics: bool = False
) -> Dict:
    """
    Main algorithm using temporal stability for super-community detection.
    
    This algorithm combines:
    1. Super-community detection using temporal stability of UASE embeddings
    2. Sub-community detection within each super-community using dynamic DC-SBM methods
    
    Parameters
    ----------
    adjacency_matrices : List[np.ndarray]
        List of T adjacency matrices, each n x n
    D : int, optional
        Initial UASE dimension. If None, use 20.
    R_super : int, optional
        Number of super-communities. If None, estimated automatically.
    d_subgraph : int, optional
        Subgraph embedding dimension. If None, estimated per subgraph.
    R_subgraph : List[int], optional
        Number of communities per subgraph. If None, estimated per subgraph.
    significance_level : float
        Significance level for stable dimension detection (default 0.01)
    regularization_strength : float
        Regularization for AIC-based dimension estimation (default 3.1)
    verbose : bool
        Print detailed progress
    plot_diagnostics : bool
        Show diagnostic plots
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'super_communities': Static super-community assignments dim n
        - 'communities': Dynamic community assignments dim Txn
        - 'hierarchical_labels': List of (super, sub) tuples per time dim Txn
        - 'R_super_est': Estimated number of super-communities
        - 'd_est_per_subgraph': Dict of embedding dimensions per subgraph (can remove) 
        - 'R_est_per_subgraph': Dict of community counts per subgraph (can remove) 
        - 'super_community_analytics': Analytics from super-community detection (can remove) 
        - 'subgraph_results': Detailed results per subgraph (can remove) 
    """
    n = adjacency_matrices[0].shape[0]
    T = len(adjacency_matrices)
    
    if verbose:
        print("="*70)
        print("DYNAMIC HIERARCHICAL DC-SBM COMMUNITY DETECTION")
        print("(Using Temporal Stability Method)")
        print("="*70)
        print(f"Data: {n} nodes, {T} time points")
        edge_density = np.mean([np.sum(A)/(n*(n-1)) for A in adjacency_matrices])
        print(f"Edge density: {edge_density:.4f}")
    
    # =========================================================================
    # STEPS 1-4: SUPER-COMMUNITY DETECTION USING TEMPORAL STABILITY
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("SUPER-COMMUNITY DETECTION")
        print("="*70)
    
    # Use default D if not provided
    if D is None:
        D = 20
        if verbose:
            print(f"Using default initial dimension D = {D}")
    
    # Detect super-communities using temporal stability
    super_community_labels, super_analytics = detect_super_communities_stable(
        adjacency_matrices,
        D_init=D,
        significance_level=significance_level,
        n_super_communities=R_super,
        verbose=verbose
    )
    
    R_super_est = len(np.unique(super_community_labels))
    
    if plot_diagnostics:
        # Plot stability scores and cutoff
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Stability scores
        scores = super_analytics['sorted_scores']
        n_stable = super_analytics['n_stable']
        ax1.plot(range(len(scores)), scores, 'o-', markersize=8)
        ax1.axvline(x=n_stable-0.5, color='red', linestyle='--', 
                   label=f'Cutoff ({n_stable} dims)')
        ax1.set_xlabel('Dimension Rank')
        ax1.set_ylabel('Stability Score')
        ax1.set_title('Temporal Stability Scores')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: First two stable dimensions
        Y_stable_0 = super_analytics['Y_stable_list'][0]
        if Y_stable_0.shape[1] >= 2:
            for k in range(R_super_est):
                mask = super_community_labels == k
                ax2.scatter(Y_stable_0[mask, 0], Y_stable_0[mask, 1], 
                          label=f'SC {k}', alpha=0.6, s=20)
            ax2.set_xlabel('Stable Dimension 1')
            ax2.set_ylabel('Stable Dimension 2')
            ax2.set_title('Super-Communities in Stable Subspace')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # =========================================================================
    # STEP 5: PROCESS EACH SUPER-COMMUNITY SUBGRAPH
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("SUB-COMMUNITY DETECTION WITHIN SUPER-COMMUNITIES")
        print("="*70)
    
    # Initialize storage for results
    d_est_per_subgraph = {}
    R_est_per_subgraph = {}
    subgraph_results = {}
    
    # Create index mapper
    from dyn_hdcsbm import SubgraphIndexMapper
    mapper = SubgraphIndexMapper(super_community_labels)
    
    for sg_id in range(R_super_est):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing Super-Community {sg_id}")
        
        # Get subgraph nodes
        sg_nodes = mapper.get_subgraph_nodes(sg_id)
        n_sg = len(sg_nodes)
        
        if n_sg < 10:  # Skip if too small
            if verbose:
                print(f"  Skipping: only {n_sg} nodes")
            d_est_per_subgraph[sg_id] = 1
            R_est_per_subgraph[sg_id] = 1
            subgraph_results[sg_id] = {
                'labels': np.zeros((T, n_sg), dtype=int),
                'message': 'Subgraph too small'
            }
            continue
        
        # Extract subgraph adjacency matrices
        A_subgraph_list = []
        for t in range(T):
            A_t = adjacency_matrices[t]
            A_sg = A_t[np.ix_(sg_nodes, sg_nodes)]
            A_subgraph_list.append(A_sg)
        
        # Step 5.1: Estimate embedding dimension for subgraph
        if d_subgraph is None:
            if verbose:
                print(f"\n  Step 5.{sg_id}.1: Estimating embedding dimension")
            
            d_sg_est, d_info = estimate_d_regularized_aic(
                A_subgraph_list,
                d_max=min(20, n_sg//2),
                regularization_strength=regularization_strength,
                verbose=False
            )
            d_est_per_subgraph[sg_id] = d_sg_est
            
            if verbose:
                print(f"    Estimated d = {d_sg_est}")
        else:
            d_sg_est = d_subgraph
            d_est_per_subgraph[sg_id] = d_sg_est
        
        # Step 5.2: Estimate number of communities in subgraph
        if R_subgraph is None:
            if verbose:
                print(f"\n  Step 5.{sg_id}.2: Estimating number of communities")
            
            # Get UASE embeddings for subgraph
            Y_sg_list = unfolded_adjacency_spectral_embedding(A_subgraph_list, d=d_sg_est)
            
            # Convert to spherical if possible
            if d_sg_est >= 2:
                Y_sg_spherical = [cartesian_to_spherical(Y) for Y in Y_sg_list]
            else:
                Y_sg_spherical = Y_sg_list
            
            R_sg_est, R_info = estimate_R_silhouette(
                Y_sg_spherical,
                R_max=min(10, n_sg//10),
                # stability_weight=stability_weight,
                verbose=False
            )
            R_est_per_subgraph[sg_id] = R_sg_est
            
            if verbose:
                print(f"    Estimated R = {R_sg_est}")
        else:
            R_sg_est = R_subgraph[sg_id] if isinstance(R_subgraph, list) else R_subgraph
            R_est_per_subgraph[sg_id] = R_sg_est
        
        # Step 5.3: Perform dynamic community detection
        if verbose:
            print(f"\n  Step 5.{sg_id}.3: Dynamic community detection")
        
        sg_labels, sg_info = dynamic_dcsbm_clustering(
            A_subgraph_list,
            d=d_sg_est,
            R=R_sg_est
        )
        
        subgraph_results[sg_id] = {
            'labels': sg_labels,  # Shape (T, n_sg)
            'info': sg_info,
            'd_est': d_sg_est,
            'R_est': R_sg_est,
            'nodes': sg_nodes
        }
        
        if verbose:
            # Print community sizes at t=0
            labels_t0 = sg_labels[0]
            print(f"    Community sizes at t=0:")
            for comm in range(R_sg_est):
                size = np.sum(labels_t0 == comm)
                print(f"      Community {comm}: {size} nodes")
    
    # =========================================================================
    # STEP 6: COMBINE RESULTS
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("COMBINING RESULTS")
        print("-"*70)
    
    # Create full community labels array
    community_labels = np.zeros((T, n), dtype=int)
    hierarchical_labels = []
    
    for t in range(T):
        labels_t = []
        for i in range(n):
            sg_id = super_community_labels[i]
            
            if sg_id in subgraph_results and 'labels' in subgraph_results[sg_id]:
                # Get local index within subgraph
                sg_nodes = subgraph_results[sg_id]['nodes']
                local_idx = np.where(sg_nodes == i)[0][0]
                
                # Get community within subgraph
                comm_id = subgraph_results[sg_id]['labels'][t, local_idx]
                
                # Create global community ID
                offset = sum(R_est_per_subgraph[j] for j in range(sg_id))
                global_comm_id = offset + comm_id
                
                community_labels[t, i] = global_comm_id
                labels_t.append((sg_id, comm_id))
            else:
                # Handle edge cases
                community_labels[t, i] = -1
                labels_t.append((sg_id, -1))
        
        hierarchical_labels.append(labels_t)
    
    # =========================================================================
    # COMPILE FINAL RESULTS
    # =========================================================================
    results = {
        'super_communities': super_community_labels,
        'communities': community_labels,
        'hierarchical_labels': hierarchical_labels,
        'R_super_est': R_super_est,
        'd_est_per_subgraph': d_est_per_subgraph,
        'R_est_per_subgraph': R_est_per_subgraph,
        'super_community_analytics': super_analytics,
        'subgraph_results': subgraph_results
    }
    
    if verbose:
        print("\n" + "="*70)
        print("ALGORITHM COMPLETE")
        print("="*70)
        print(f"Super-communities: {R_super_est}")
        print(f"Total communities: {sum(R_est_per_subgraph.values())}")
        print(f"Hierarchical structure:")
        for sg_id in range(R_super_est):
            print(f"  Super-community {sg_id}: {R_est_per_subgraph.get(sg_id, 0)} communities")
    
    return results
