"""
Tools and methods for simulating dynamic degree-corrected stochastic block models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict


def generate_adjacency_from_probability(P: np.ndarray) -> np.ndarray:
    """
    Generate adjacency matrix A from probability matrix P.
    
    Parameters:
    -----------
    P : np.ndarray
        Probability matrix with entries in [0,1]. Can be symmetric or just lower triangular.
        
    Returns:
    --------
    A : np.ndarray
        Binary adjacency matrix where A[i,j] = 1 with probability P[i,j].
        Diagonal is set to 0, and matrix is symmetric.
    """
    n = P.shape[0]
    assert P.shape == (n, n), "P must be square"
    assert np.all((P >= 0) & (P <= 1)), "P entries must be in [0,1]"
    
    U = np.random.uniform(0, 1, size=(n, n))
    
    # Create adjacency matrix
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if U[i, j] < P[i, j]:
                A[i, j] = 1
                A[j, i] = 1  # Force symmetry 
    
    # Ensure diagonal is zero
    np.fill_diagonal(A, 0)
    
    return A


class LatentPositionGenerator(ABC):
    """
    Abstract base class for generating latent positions over time.
    All generators must know n, d, and T at construction time.
    """
    
    def __init__(self, n: int, d: int, T: int):
        """
        Initialize generator with fixed dimensions.
        
        Parameters:
        -----------
        n : int
            Number of nodes
        d : int
            Dimension of latent space
        T : int
            Time horizon (t = 0, 1, ..., T)
        """
        self.n = n
        self.d = d
        self.T = T
    
    @abstractmethod
    def get_X(self, t: int) -> np.ndarray:
        """
        Get latent position matrix at time t.
        
        Parameters:
        -----------
        t : int
            Time point (must be in [0, T])
            
        Returns:
        --------
        X : np.ndarray
            n x d latent position matrix
        """
        pass


class DynamicRDPG:
    """
    Dynamic Random Dot Product Graph model.
    Generates networks from time-varying latent positions.
    """
    
    def __init__(self, generator: LatentPositionGenerator):
        """
        Initialize Dynamic RDPG with a latent position generator.
        
        Parameters:
        -----------
        generator : LatentPositionGenerator
            Object that generates n x d latent positions for each time t
        """
        self.generator = generator
        self.n = generator.n
        self.d = generator.d
        self.T = generator.T
    
    def get_X(self, t: int) -> np.ndarray:
        """Get latent positions at time t."""
        if t > self.T:
            raise ValueError(f"Time {t} exceeds horizon T={self.T}")
        return self.generator.get_X(t)
    
    def get_probability_matrix(self, t: int) -> np.ndarray:
        """Get probability matrix P = X @ X.T at time t."""
        X = self.get_X(t)
        P = X @ X.T
        return np.clip(P, 0, 1)
    
    def generate_adjacency_matrix(self, t: int) -> np.ndarray:
        """Generate adjacency matrix at time t."""
        P = self.get_probability_matrix(t)
        return generate_adjacency_from_probability(P)
    
    def generate_adjacency_matrices(self, time_points: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Generate adjacency matrices for specified time points.
        
        Parameters:
        -----------
        time_points : List[int], optional
            Time points to generate. If None, generates for all t in [0, T]
            
        Returns:
        --------
        List[np.ndarray]
            List of adjacency matrices
        """
        if time_points is None:
            time_points = list(range(self.T + 1))
        
        return [self.generate_adjacency_matrix(t) for t in time_points]


class FixedLatentPositionGenerator(LatentPositionGenerator):
    """
    Generator for fixed latent positions (static RDPG).
    """
    
    def __init__(self, X: np.ndarray):
        """
        Initialize with fixed latent positions.
        
        Parameters:
        -----------
        X : np.ndarray
            n x d latent position matrix
        """
        n, d = X.shape
        super().__init__(n, d, T=0)  # T=0 for static case
        self.X = X
    
    def get_X(self, t: int) -> np.ndarray:
        """Return the fixed latent positions regardless of t."""
        return self.X.copy()


class MarkovSBMGenerator(LatentPositionGenerator):
    """
    Latent position generator for Markov Dynamic SBM.
    Node assignments evolve via Markov chain, positions come from community centers.
    """
    
    def __init__(self, n: int, d: int, T: int,
                 community_centers: np.ndarray,
                 community_probs: np.ndarray,
                 stay_prob: float,
                 random_state: Optional[int] = None):
        """
        Initialize Markov SBM generator.
        
        Parameters:
        -----------
        n : int
            Number of nodes
        d : int
            Dimension of latent space
        T : int
            Time horizon
        community_centers : np.ndarray
            R x d matrix of community centers
        community_probs : np.ndarray
            Probability distribution over communities
        stay_prob : float
            Probability of staying in current community
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(n, d, T)
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.community_centers = community_centers
        self.R, self.d_centers = community_centers.shape
        assert self.d_centers == d, f"Community centers dimension {self.d_centers} != d={d}"
        
        self.community_probs = community_probs
        self.stay_prob = stay_prob
        
        # Validate inputs
        assert len(community_probs) == self.R
        assert np.all(community_probs >= 0) and np.all(community_probs <= 1)
        assert np.abs(np.sum(community_probs) - 1.0) < 1e-10
        assert 0 <= stay_prob <= 1
        
        # Pre-generate community assignments
        self.I = np.zeros((T + 1, n), dtype=int)
        
        # Initialize at t=0
        self.I[0] = np.random.choice(self.R, size=n, p=community_probs)
        
        # Generate for t=1 to T
        for t in range(1, T + 1):
            for i in range(n):
                if np.random.rand() < stay_prob:
                    self.I[t, i] = self.I[t - 1, i]
                else:
                    self.I[t, i] = np.random.choice(self.R, p=community_probs)
    
    def get_X(self, t: int) -> np.ndarray:
        """Get latent positions at time t."""
        X = np.zeros((self.n, self.d))
        for i in range(self.n):
            X[i] = self.community_centers[self.I[t, i]]
        return X
    
    def get_assignment(self, t: int, i: int) -> int:
        """Get community assignment for node i at time t."""
        return self.I[t, i]
    
    def get_community_members(self, t: int, j: int) -> np.ndarray:
        """Get all nodes in community j at time t."""
        return np.where(self.I[t] == j)[0]
    
    def get_trajectory(self, i: int) -> np.ndarray:
        """Get community trajectory for node i."""
        return self.I[:, i]


class MarkovDegreeCorrectedSBMGenerator(LatentPositionGenerator):
    """
    Latent position generator for Markov Dynamic Degree-Corrected SBM.
    Node assignments evolve via Markov chain, positions are weighted community centers.
    """
    
    def __init__(self, n: int, d: int, T: int,
                 community_centers: np.ndarray,
                 community_probs: np.ndarray,
                 stay_prob: float,
                 alpha: float = 8.0,
                 beta: float = 2.0,
                 random_state: Optional[int] = None):
        """
        Initialize Markov Degree-Corrected SBM generator.
        
        Parameters:
        -----------
        n : int
            Number of nodes
        d : int
            Dimension of latent space
        T : int
            Time horizon
        community_centers : np.ndarray
            R x d matrix of community centers
        community_probs : np.ndarray
            Probability distribution over communities
        stay_prob : float
            Probability of staying in current community
        alpha : float
            Beta distribution alpha parameter (default: 8.0)
        beta : float
            Beta distribution beta parameter (default: 2.0)
            Note: Default alpha=8, beta=2 gives mean weight of 0.8
        random_state : int, optional
            Random seed for reproducibility
        """
        super().__init__(n, d, T)
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.community_centers = community_centers
        self.R, self.d_centers = community_centers.shape
        assert self.d_centers == d, f"Community centers dimension {self.d_centers} != d={d}"
        
        self.community_probs = community_probs
        self.stay_prob = stay_prob
        self.alpha = alpha
        self.beta_param = beta  # Renamed to avoid conflict with beta function
        
        # Validate inputs
        assert len(community_probs) == self.R
        assert np.all(community_probs >= 0) and np.all(community_probs <= 1)
        assert np.abs(np.sum(community_probs) - 1.0) < 1e-10
        assert 0 <= stay_prob <= 1
        assert alpha > 0 and beta > 0
        
        # Pre-generate community assignments (same as MarkovSBM)
        self.I = np.zeros((T + 1, n), dtype=int)
        
        # Initialize at t=0
        self.I[0] = np.random.choice(self.R, size=n, p=community_probs)
        
        # Generate for t=1 to T
        for t in range(1, T + 1):
            for i in range(n):
                if np.random.rand() < stay_prob:
                    self.I[t, i] = self.I[t - 1, i]
                else:
                    self.I[t, i] = np.random.choice(self.R, p=community_probs)
        
        # Pre-generate weights for all time steps
        self.W = np.random.beta(alpha, beta, size=(T + 1, n))
    
    def get_X(self, t: int) -> np.ndarray:
        """Get weighted latent positions at time t."""
        X = np.zeros((self.n, self.d))
        for i in range(self.n):
            j = self.I[t, i]  # Community assignment
            w = self.W[t, i]  # Weight at time t
            X[i] = w * self.community_centers[j]
        return X
    
    def get_assignment(self, t: int, i: int) -> int:
        """Get community assignment for node i at time t."""
        return self.I[t, i]
    
    def get_weight(self, t: int, i: int) -> float:
        """Get weight for node i at time t."""
        return self.W[t, i]
    
    def get_community_members(self, t: int, j: int) -> np.ndarray:
        """Get all nodes in community j at time t."""
        return np.where(self.I[t] == j)[0]
    
    def get_trajectory(self, i: int) -> Dict[str, np.ndarray]:
        """Get community and weight trajectory for node i."""
        return {
            'communities': self.I[:, i],
            'weights': self.W[:, i]
        }
    
    def get_mean_weight(self) -> float:
        """Get the expected mean weight from Beta distribution."""
        return self.alpha / (self.alpha + self.beta_param)


class RotatingLatentPositionGenerator(LatentPositionGenerator):
    """
    Example generator: latent positions that rotate over time.
    """
    
    def __init__(self, n: int, d: int, T: int, base_positions: Optional[np.ndarray] = None):
        """
        Initialize rotating generator.
        
        Parameters:
        -----------
        n : int
            Number of nodes
        d : int
            Dimension (must be >= 2 for rotation)
        T : int
            Time horizon
        base_positions : np.ndarray, optional
            Initial positions. If None, generates random positions.
        """
        super().__init__(n, d, T)
        
        if base_positions is None:
            # Generate random normalized positions
            base_positions = np.random.randn(n, d)
            base_positions = base_positions / np.linalg.norm(base_positions, axis=1, keepdims=True)
        
        assert base_positions.shape == (n, d)
        self.base_positions = base_positions
    
    def get_X(self, t: int) -> np.ndarray:
        """Get rotated positions at time t."""
        if self.d < 2:
            return self.base_positions.copy()
        
        # Simple rotation in first two dimensions
        theta = t * 0.1
        rotation = np.eye(self.d)
        rotation[0, 0] = np.cos(theta)
        rotation[0, 1] = -np.sin(theta)
        rotation[1, 0] = np.sin(theta)
        rotation[1, 1] = np.cos(theta)
        
        return self.base_positions @ rotation.T


# Convenience factory functions
def create_static_rdpg(X: np.ndarray) -> DynamicRDPG:
    """Create a static RDPG (T=0) from fixed latent positions."""
    generator = FixedLatentPositionGenerator(X)
    return DynamicRDPG(generator)


def create_static_sbm(n: int, community_centers: np.ndarray, 
                     community_probs: np.ndarray) -> DynamicRDPG:
    """Create a static SBM (T=0)."""
    d = community_centers.shape[1]
    generator = MarkovSBMGenerator(n, d, T=0, 
                                  community_centers=community_centers,
                                  community_probs=community_probs,
                                  stay_prob=1.0)  # No transitions for static
    return DynamicRDPG(generator)


def create_markov_sbm(n: int, T: int, community_centers: np.ndarray,
                     community_probs: np.ndarray, stay_prob: float,
                     ) -> DynamicRDPG:
    """Create a Markov Dynamic SBM."""
    d = community_centers.shape[1]
    generator = MarkovSBMGenerator(n, d, T, community_centers,
                                  community_probs, stay_prob)
    return DynamicRDPG(generator)


def create_markov_dc_sbm(n: int, T: int, community_centers: np.ndarray,
                        community_probs: np.ndarray, stay_prob: float,
                        alpha: float = 8.0, beta: float = 2.0,
                        random_state: Optional[int] = None) -> DynamicRDPG:
    """Create a Markov Dynamic Degree-Corrected SBM."""
    d = community_centers.shape[1]
    generator = MarkovDegreeCorrectedSBMGenerator(
        n, d, T, community_centers, community_probs, stay_prob,
        alpha, beta, random_state)
    return DynamicRDPG(generator)


# Example usage
if __name__ == "__main__":
    # Test 1: Static RDPG
    print("Test 1 - Static RDPG:")
    n, d = 10, 3
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    static_rdpg = create_static_rdpg(X)
    A = static_rdpg.generate_adjacency_matrix(0)
    print(f"Generated adjacency matrix with {np.sum(A) // 2} edges\n")
    
    # Test 2: Static SBM
    print("Test 2 - Static SBM:")
    R = 3  # Number of communities
    community_centers = np.array([
        [0.8, 0.1],
        [0.1, 0.8],
        [0.4, 0.4]
    ])
    community_probs = np.array([0.4, 0.4, 0.2])
    n = 30
    
    static_sbm = create_static_sbm(n, community_centers, community_probs)
    A_sbm = static_sbm.generate_adjacency_matrix(0)
    print(f"Generated SBM with {np.sum(A_sbm) // 2} edges")
    
    # Access generator to get community info
    sbm_gen = static_sbm.generator
    for j in range(R):
        members = sbm_gen.get_community_members(0, j)
        print(f"  Community {j}: {len(members)} nodes")
    print()
    
    # Test 3: Dynamic RDPG with rotation
    print("Test 3 - Dynamic RDPG with rotation:")
    n, d, T = 20, 3, 5
    rotating_gen = RotatingLatentPositionGenerator(n, d, T)
    dynamic_rdpg = DynamicRDPG(rotating_gen)
    
    adjacency_matrices = dynamic_rdpg.generate_adjacency_matrices()
    for t, A in enumerate(adjacency_matrices):
        print(f"  Time {t}: {np.sum(A) // 2} edges")
    print()
    
    # Test 4: Markov SBM
    print("Test 4 - Markov Dynamic SBM:")
    n, T = 50, 10
    stay_prob = 0.8
    
    markov_sbm = create_markov_sbm(n, T, community_centers, community_probs, 
                                   stay_prob)
    
    print(f"Stay probability: {stay_prob}")
    print(f"Time horizon: {T}")
    
    # Track community sizes
    time_points = [0, 5, 10]
    for t in time_points:
        sizes = []
        for j in range(R):
            members = markov_sbm.generator.get_community_members(t, j)
            sizes.append(len(members))
        print(f"  Time {t}: Community sizes = {sizes}")
    
    # Show node trajectories
    print("\nNode trajectories (first 3 nodes):")
    for i in range(3):
        traj = markov_sbm.generator.get_trajectory(i)
        switches = sum(1 for t in range(1, len(traj)) if traj[t] != traj[t-1])
        print(f"  Node {i}: {traj} (switched {switches} times)")
    print()
    
    # Test 5: Markov Degree-Corrected SBM
    print("Test 5 - Markov Dynamic Degree-Corrected SBM:")
    markov_dc_sbm = create_markov_dc_sbm(n, T, community_centers, community_probs,
                                        stay_prob, alpha=8.0, beta=2.0)
    
    dc_gen = markov_dc_sbm.generator
    print(f"Expected mean weight: {dc_gen.get_mean_weight():.3f}")
    
    # Compare edge counts
    for t in [0, 5, 10]:
        A_standard = markov_sbm.generate_adjacency_matrix(t)
        A_dc = markov_dc_sbm.generate_adjacency_matrix(t)
        print(f"  Time {t}: Standard = {np.sum(A_standard)//2} edges, "
              f"DC = {np.sum(A_dc)//2} edges")
    
    # Show weight variation
    print("\nWeight statistics (first 5 nodes):")
    for i in range(5):
        traj = dc_gen.get_trajectory(i)
        weights = traj['weights']
        print(f"  Node {i}: mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")