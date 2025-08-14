import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import from the dynamic RDPG file
from dyn_dcsbm import (
    DynamicRDPG,
    LatentPositionGenerator,
    MarkovDegreeCorrectedSBMGenerator
)


class SubgraphIndexMapper:
    """
    Maps between global indices and subgraph-local indices.  

    The reason for this is that both main graph and sub graph need to 
    map nodes to 0..n or 0..m, but we need to remember which node in the 
    main graph corresponds to node i in the subgraph, and vice versa.  
    This class translates between the two.  
    
    Given an assignment of n nodes to R subgraphs, provides:
    - Mapping from global index i to (subgraph_id, local_index)
    - Mapping from (subgraph_id, local_index) to global index
    - Utilities for combining/splitting matrices
    """
    
    def __init__(self, assignments: np.ndarray):
        """
        Initialize mapper with pre-existing subgraph assignments.
        
        Parameters:
        -----------
        assignments : np.ndarray
            Array of length n where entry i is the subgraph ID 
            for node i.  Ie an input [0,0,0,1,1,2] means that we have three subgraphs and that
            node 0,1,2 are in subgraph 0, node 3,4 are in subgraph 1, and node 5 is in subgraph 2.  

            We assume the labelling is contiguous, ie input like [0,0,0,2,2,3] is not allowed.  
        """
        self.assignments = assignments
        self.n = len(assignments)
        self.subgraph_ids = np.unique(assignments)
        self.n_subgraphs = len(self.subgraph_ids)
        
        # Check if subgraph IDs are contiguous starting from 0
        if not np.array_equal(self.subgraph_ids, np.arange(self.n_subgraphs)):
            raise ValueError(f"Subgraph IDs must be contiguous from 0 to {self.n_subgraphs-1}, "
                           f"got {self.subgraph_ids}")
        
        # Build mappings
        self.global_to_local = {}
        self.local_to_global = {sg_id: [] for sg_id in range(self.n_subgraphs)}
        
        for i, sg_id in enumerate(assignments):
            local_idx = len(self.local_to_global[sg_id])
            self.global_to_local[i] = (sg_id, local_idx)
            self.local_to_global[sg_id].append(i)
        
        # Convert lists to arrays and store sizes
        for sg_id in range(self.n_subgraphs):
            self.local_to_global[sg_id] = np.array(self.local_to_global[sg_id])
        
        self.subgraph_sizes = {sg_id: len(nodes) for sg_id, nodes in self.local_to_global.items()}
    
    def get_local_index(self, global_idx: int) -> Tuple[int, int]:
        """Map global index to (subgraph_id, local_index)."""
        if not 0 <= global_idx < self.n:
            raise ValueError(f"Global index {global_idx} out of range [0, {self.n-1}]")
        return self.global_to_local[global_idx]
    
    def get_global_index(self, subgraph_id: int, local_idx: int) -> int:
        """Map (subgraph_id, local_index) to global index."""
        if subgraph_id not in self.subgraph_ids:
            raise ValueError(f"Invalid subgraph ID {subgraph_id}")
        if not 0 <= local_idx < self.subgraph_sizes[subgraph_id]:
            raise ValueError(f"Local index {local_idx} out of range for subgraph {subgraph_id}")
        return self.local_to_global[subgraph_id][local_idx]
    
    def get_subgraph_nodes(self, subgraph_id: int) -> np.ndarray:
        """Get all global indices for nodes in a subgraph."""
        if subgraph_id not in self.subgraph_ids:
            raise ValueError(f"Invalid subgraph ID {subgraph_id}")
        return self.local_to_global[subgraph_id].copy()
    
    def get_subgraph_size(self, subgraph_id: int) -> int:
        """Get number of nodes in a subgraph."""
        if subgraph_id not in self.subgraph_ids:
            raise ValueError(f"Invalid subgraph ID {subgraph_id}")
        return self.subgraph_sizes[subgraph_id]
    
    def combine_subgraph_matrices(self, subgraph_matrices: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Combine subgraph matrices into a single global matrix.
        
        Parameters:
        -----------
        subgraph_matrices : dict
            Dictionary mapping subgraph_id -> matrix of shape (K_j, d)
            
        Returns:
        --------
        np.ndarray
            Combined matrix of shape (n, d)
        """
        if not subgraph_matrices:
            raise ValueError("No subgraph matrices provided")
        
        # Validate and get dimension
        d = None
        for sg_id, matrix in subgraph_matrices.items():
            if sg_id not in self.subgraph_ids:
                raise ValueError(f"Invalid subgraph ID {sg_id} in input")
            if matrix.shape[0] != self.subgraph_sizes[sg_id]:
                raise ValueError(f"Subgraph {sg_id} matrix has {matrix.shape[0]} rows, "
                               f"expected {self.subgraph_sizes[sg_id]}")
            if d is None:
                d = matrix.shape[1]
            elif matrix.shape[1] != d:
                raise ValueError(f"All subgraph matrices must have same number of columns")
        
        # Create and fill global matrix
        global_matrix = np.zeros((self.n, d))
        for sg_id, matrix in subgraph_matrices.items():
            global_matrix[self.local_to_global[sg_id]] = matrix
        
        return global_matrix
    
    def split_global_matrix(self, global_matrix: np.ndarray) -> Dict[int, np.ndarray]:
        """Split a global matrix into subgraph matrices."""
        if global_matrix.shape[0] != self.n:
            raise ValueError(f"Global matrix has {global_matrix.shape[0]} rows, expected {self.n}")
        
        return {sg_id: global_matrix[nodes].copy() 
                for sg_id, nodes in self.local_to_global.items()}


@dataclass
class SubgraphDCSBMParameters:
    """Parameters for a single subgraph's DC-SBM.
    
    NB we provide explicit community center vectors.  
    """
    community_centers: np.ndarray  # R_j x d matrix
    community_probs: np.ndarray    # R_j probabilities
    stay_prob: float               # Markov stay probability
    alpha: float = 8.0             # Beta distribution alpha
    beta: float = 2.0              # Beta distribution beta


class HierarchicalMarkovDCSBMGenerator(LatentPositionGenerator):
    """
    Hierarchical Markov Dynamic Degree-Corrected SBM.
    
    Nodes are statically assigned to subgraphs.
    Each subgraph has its own Markov DC-SBM dynamics.
    """
    
    def __init__(self, 
                 n: int,
                 d: int, 
                 T: int,
                 subgraph_assignments: np.ndarray,
                 subgraph_params: List[SubgraphDCSBMParameters],
                 random_state: Optional[int] = None):
        """
        Initialize hierarchical generator.
        
        Parameters:
        -----------
        n : int
            Total number of nodes
        d : int
            Dimension of latent space
        T : int
            Time horizon
        subgraph_assignments : np.ndarray
            Array of length n with subgraph assignments (0-indexed, contiguous)
        subgraph_params : List[SubgraphDCSBMParameters]
            Parameters for each subgraph's DC-SBM
        random_state : int, optional
            Random seed
        """
        super().__init__(n, d, T)
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Create index mapper
        self.mapper = SubgraphIndexMapper(subgraph_assignments)
        self.n_subgraphs = self.mapper.n_subgraphs
        
        # Validate parameters
        if len(subgraph_params) != self.n_subgraphs:
            raise ValueError(f"Expected {self.n_subgraphs} parameter sets, got {len(subgraph_params)}")
        
        for i, params in enumerate(subgraph_params):
            if params.community_centers.shape[1] != d:
                raise ValueError(f"Subgraph {i} centers have dimension {params.community_centers.shape[1]}, expected {d}")
        
        self.subgraph_params = subgraph_params
        
        # Create generators for each subgraph
        self.subgraph_generators = {}
        for sg_id in range(self.n_subgraphs):
            n_sg = self.mapper.get_subgraph_size(sg_id)
            if n_sg > 0:  # Only create if subgraph is non-empty
                params = subgraph_params[sg_id]
                self.subgraph_generators[sg_id] = MarkovDegreeCorrectedSBMGenerator(
                    n=n_sg,
                    d=d,
                    T=T,
                    community_centers=params.community_centers,
                    community_probs=params.community_probs,
                    stay_prob=params.stay_prob,
                    alpha=params.alpha,
                    beta=params.beta,
                    random_state=random_state
                )
    
    def get_X(self, t: int) -> np.ndarray:
        """
        Get latent positions at time t by combining subgraph positions.
        
        Parameters:
        -----------
        t : int
            Time point
            
        Returns:
        --------
        X : np.ndarray
            n x d latent position matrix
        """
        # Get positions from each subgraph
        subgraph_matrices = {}
        for sg_id, generator in self.subgraph_generators.items():
            subgraph_matrices[sg_id] = generator.get_X(t)
        
        # Combine using the mapper
        return self.mapper.combine_subgraph_matrices(subgraph_matrices)
    
    def get_hierarchical_assignment(self, t: int, i: int) -> Tuple[int, int, float]:
        """
        Get complete hierarchical information for node i at time t.
        
        Parameters:
        -----------
        t : int
            Time point
        i : int
            Global node index
            
        Returns:
        --------
        (subgraph_id, community_id, weight) : tuple
            Subgraph assignment, community within subgraph, and DC weight
        """
        sg_id, local_idx = self.mapper.get_local_index(i)
        
        if sg_id in self.subgraph_generators:
            gen = self.subgraph_generators[sg_id]
            community_id = gen.get_assignment(t, local_idx)
            weight = gen.get_weight(t, local_idx)
        else:
            community_id = 0
            weight = 1.0
        
        return sg_id, community_id, weight
    
    def get_subgraph_assignment(self, i: int) -> int:
        """Get which subgraph node i belongs to (static)."""
        sg_id, _ = self.mapper.get_local_index(i)
        return sg_id
    
    def get_trajectory(self, i: int) -> Dict:
        """
        Get full trajectory for node i.
        
        Returns:
        --------
        dict with keys:
            'subgraph': int (static)
            'communities': array of community assignments over time
            'weights': array of DC weights over time
        """
        sg_id, local_idx = self.mapper.get_local_index(i)
        
        result = {
            'subgraph': sg_id,
            'communities': np.zeros(self.T + 1, dtype=int),
            'weights': np.ones(self.T + 1)
        }
        
        if sg_id in self.subgraph_generators:
            local_traj = self.subgraph_generators[sg_id].get_trajectory(local_idx)
            result['communities'] = local_traj['communities']
            result['weights'] = local_traj['weights']
        
        return result


class HierarchicalDynamicDCSBM(DynamicRDPG):
    """
    Hierarchical Dynamic DC-SBM.  

    Feed in a HierarchicalMarkovDCSBMGenerator generator and the dynamic RDPG 
    model can be used to calculate adjacency matrices at each time point.  
    """
    
    def __init__(self, generator: HierarchicalMarkovDCSBMGenerator):
        super().__init__(generator)
        self.hierarchical_generator = generator
    

# Factory function to create the hierarchical model.  
def create_hierarchical_markov_dcsbm(
        subgraph_assignments: np.ndarray,
        T: int,
        subgraph_params: List[SubgraphDCSBMParameters],
        random_state: Optional[int] = None) -> HierarchicalDynamicDCSBM:
    """
    Create a Hierarchical Markov Dynamic DC-SBM, given a static subgraph/super-community assignment.  
    NB, hidden in the subgraph parameters is the shape of the community vectors, ie the embedding dimension used 
    
    Parameters:
    -----------
    subgraph_assignments : np.ndarray
        Pre-existing assignment of nodes to subgraphs
    T : int
        Time horizon
    subgraph_params : List[SubgraphDCSBMParameters]
        Parameters for each subgraph
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    HierarchicalDynamicDCSBM
    """
    n = len(subgraph_assignments)
    d = subgraph_params[0].community_centers.shape[1]
    
    generator = HierarchicalMarkovDCSBMGenerator(
        n=n,
        d=d,
        T=T,
        subgraph_assignments=subgraph_assignments,
        subgraph_params=subgraph_params,
        random_state=random_state
    )
    
    return HierarchicalDynamicDCSBM(generator)


def generate_hierarchical_community_vectors(rho1=0.05, rho2=0.1, rho3=0.3, R=[3, 2, 3]):
    """
    Generate community vectors with controlled scalar product structure.
    
    Parameters:
    -----------
    rho1 : float
        Scalar product for vectors in different super communities
    rho2 : float
        Scalar product for vectors in same super community but different sub communities
    rho3 : float
        Scalar product for vectors in same super and sub communities
    R : list of int
        Number of sub communities in each super community
    
    Returns:
    --------
    vectors : numpy array
        a matrix containing vectors obeying the required scalar product structure.  
    labels : list of tuples
        Each tuple is (super_community, sub_community) for the corresponding vector
    """

    # The algorithm works very simply.  All vectors share one nonzero coordinate of \sqrt{rho1}.  
    # Vectors in the same supergroup share another nonzero coordinate of \sqrt{rho2 - rho1}.  
    # Vectors in the same sub-supergroup share another nonzero coordinate of \sqrt{rho3 - rho2}.  
    # The remaining coordinates are zero.  

    # Validate inputs
    assert rho1 <= rho2 <= rho3, "Must have rho1 <= rho2 <= rho3"
    
    K = len(R)              # Number of super communities
    total_vectors = sum(R)  # Total number of vectors
    
    # Calculate dimension.  We use different batches of dimensions for the different levels.  
    # 1 (first coordinate) + K (super community coords) + sum(R) (sub community coords)
    dim = 1 + K + total_vectors
    
    # Initialize vectors
    vectors = np.zeros((total_vectors, dim))
    labels = []
    
    # Track current vector index
    vec_idx = 0
    # Track starting index for sub communities of each super community
    sub_comm_start_idx = 1 + K
    
    for super_idx in range(K):
        for sub_idx in range(R[super_idx]):
            # First coordinate: sqrt(rho1)
            vectors[vec_idx, 0] = np.sqrt(rho1)
            
            # Super community coordinate: sqrt(rho2 - rho1)
            vectors[vec_idx, 1 + super_idx] = np.sqrt(rho2 - rho1)
            
            # Sub community coordinate: sqrt(rho3 - rho2)
            # Calculate the global sub community index
            global_sub_idx = sub_comm_start_idx + sum(R[:super_idx]) + sub_idx
            vectors[vec_idx, global_sub_idx] = np.sqrt(rho3 - rho2)
            
            labels.append((super_idx, sub_idx))
            vec_idx += 1
    
    return vectors


def make_hierarchical_model(
    n=500, T=20, 
    rho1=0.05, rho2=0.1, rho3=0.3, 
    R=np.array([3, 2, 3]), 
    sup_probs = np.array([0.3, 0.45, 0.25]),
    subgraph_probs=[
        np.array([0.3, 0.45, 0.25]), 
        np.array([0.7,0.3]), 
        np.array([0.2, 0.3, 0.5])
    ], 
    stay_probs=np.array([0.95, 0.9, 0.85]),
    alpha=np.array([8.0, 8.0, 8.0]),
    beta=np.array([2.0, 2.0, 2.0])
):
    """
    Main convenience function to create a dynamic hierarchical model.  

    Parameters:
    -----------
    n : int
        Number of nodes
    T : int
        Time horizon
    rho1, rho2, rho3 : float
        edge connection likelihood parameters for different super-, different sub-, 
        same sub-community.  Must have rho1 <= rho2 <= rho3.  
    R : list of int
        Number of sub communities in each super community
    sup_probs : list of float
        Probabilities for assignemt to super communities
    subgraph_probs : list of list of float
        Probabilities for initial assignment to different sub communities in each super community
    stay_probs : list of float
        in [0,1].  Probability of staying in the same community at each time point.  
    alpha, beta : list of float
        Parameter list for the beta distributions per super-community, used to generate 
        the DC weights.  
    """

    community_centers = generate_hierarchical_community_vectors(rho1, rho2, rho3, R)
    subgraph_assignments = np.random.choice(len(R), size=n, p=sup_probs)
    # need index [0, R_1, R_1 + R_2, ..] to slice out the right section of the community center matrix.  
    R_idx = [0] + list(np.cumsum(R))  
    subgraph_params = [
        SubgraphDCSBMParameters(
            community_centers=community_centers[R_idx[i]:R_idx[i+1], :], 
            community_probs=subgraph_probs[i], 
            stay_prob=stay_probs[i], 
            alpha=alpha[i], beta=beta[i])
        for i in range(len(R))
    ]
    return create_hierarchical_markov_dcsbm(subgraph_assignments, T, subgraph_params)

