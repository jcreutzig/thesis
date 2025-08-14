from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from dyn_hdcsbm import HierarchicalDynamicDCSBM
from plot_functions import generate_hierarchical_colors
from sklearn.metrics import adjusted_rand_score as ari 

@dataclass
class HierarchicalCommunityStructure:
    """
    A useful wrappe for generating hierarchical community structures 
    for plotting and evaluation.  

    Provided as a convenience, it provides two interfaces to generate the structure:
    from_results: from estimation results (for a fit model)
    from_model: from a model object (for a true model)

    and then provides a unified interface.  Two hcs objects can be compared, and plotting functions 
    can be shared between estimated and actual models.  
    """
    # Core data
    super_communities: np.ndarray  # dim n 
    sub_communities: np.ndarray    # dim T x n 
    
    # Metadata
    n_nodes: int
    n_timepoints: int
    n_super: int
    n_sub_per_super: Dict[int, int]  # Max sub-communities per super-community
    
    # Color scheme - will be imported from generate_hierarchical_colors
    color_families: Dict  
    base_colors: Dict
    sub_colors: Dict[Tuple[int, int], Tuple[float, float, float]]
    
    @classmethod
    def from_results(cls, results: dict):
        """
        Create from estimation results, properly handling the hierarchical structure.
        """
        super_communities = results['super_communities']
        communities = results['communities']  # Global community IDs
        T, n = communities.shape
        
        # Determine number of super-communities
        n_super = len(np.unique(super_communities))
        
        # Convert global community IDs to local sub-community IDs
        sub_communities = np.zeros((T, n), dtype=int)
        n_sub_per_super = {}
        
        # For each super-community, create a mapping from global to local IDs
        for sc in range(n_super):
            sc_mask = super_communities == sc
            
            # Find all global community IDs that appear in this super-community
            global_comms_in_sc = set()
            for t in range(T):
                global_comms_in_sc.update(communities[t][sc_mask])
            
            # Create mapping from global to local sub-community index
            global_to_local = {gc: i for i, gc in enumerate(sorted(global_comms_in_sc))}
            n_sub_per_super[sc] = len(global_to_local)
            
            # Apply mapping
            for t in range(T):
                for node_idx in np.where(sc_mask)[0]:
                    global_comm = communities[t, node_idx]
                    sub_communities[t, node_idx] = global_to_local.get(global_comm, 0)
        
        # Generate color scheme
        max_sub = max(n_sub_per_super.values()) if n_sub_per_super else 4
        colors, base_colors, color_families = generate_hierarchical_colors(n_super, max_sub)
        
        return cls(
            super_communities=super_communities,
            sub_communities=sub_communities,
            n_nodes=n,
            n_timepoints=T,
            n_super=n_super,
            n_sub_per_super=n_sub_per_super,
            color_families=color_families,
            base_colors=base_colors,
            sub_colors=colors
        )
    
    @classmethod
    def from_model(cls, model: HierarchicalDynamicDCSBM):
        """
        Create HierarchicalCommunityStructure from a HierarchicalDynamicDCSBM model.
        
        Parameters
        ----------
        model : HierarchicalDynamicDCSBM
            The hierarchical model object
            
        Returns
        -------
        HierarchicalCommunityStructure
            The hierarchical structure extracted from the model
        """
        # Access the generator
        generator = model.hierarchical_generator
        n = generator.n
        T = generator.T
        
        # Get super-community assignments (these are the subgraph assignments)
        super_communities = np.zeros(n, dtype=int)
        for i in range(n):
            super_communities[i] = generator.get_subgraph_assignment(i)
        
        n_super = generator.n_subgraphs
        
        # Get sub-community assignments over time
        sub_communities = np.zeros((T, n), dtype=int)  
        
        # Track the number of unique sub-communities per super-community
        sub_comms_per_super = {sc: set() for sc in range(n_super)}
        
        # For each time point and node, get the community assignment
        for t in range(T):
            for i in range(n):
                sc_id, comm_id, _ = generator.get_hierarchical_assignment(t, i)
                sub_communities[t, i] = comm_id
                sub_comms_per_super[sc_id].add(comm_id)
        
        # Count max sub-communities per super-community
        n_sub_per_super = {sc: len(comms) for sc, comms in sub_comms_per_super.items()}
        
        # Generate color scheme
        max_sub = max(n_sub_per_super.values()) if n_sub_per_super else 4
        colors, base_colors, color_families = generate_hierarchical_colors(n_super, max_sub)
        
        return cls(
            super_communities=super_communities,
            sub_communities=sub_communities,
            n_nodes=n,
            n_timepoints=T,  
            n_super=n_super,
            n_sub_per_super=n_sub_per_super,
            color_families=color_families,
            base_colors=base_colors,
            sub_colors=colors
        )

    def get_node_color(
            self, 
            node_idx: int, 
            time_idx: int, 
            use_base: bool = False
        ) -> Tuple[float, float, float]:
        """Get RGB color for a specific node at a specific time."""
        sc = self.super_communities[node_idx]
        
        if use_base:
            return self.base_colors[sc]
        
        sub = self.sub_communities[time_idx, node_idx]
        max_sub = min(sub, 6)  # only 6 sub-community colors supported (0-5)
        
        return self.sub_colors.get((sc, max_sub), (0.8, 0.8, 0.8))

    def get_hierarchical_label(self, node_idx: int, time_idx: int) -> Tuple[int, int]:
        """Get (super-community, sub-community) tuple."""
        return (self.super_communities[node_idx], 
                self.sub_communities[time_idx, node_idx])

    def to_color_matrix(self, sample_idx: Optional[np.ndarray] = None,
                        time_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Create a color matrix for visualization.
        Returns RGB array of shape (n_nodes_display, n_time_display, 3).
        """
        if sample_idx is None:
            sample_idx = np.arange(self.n_nodes)
        
        if time_range is None:
            t_start, t_end = 0, self.n_timepoints
        else:
            t_start, t_end = time_range
        
        n_display = len(sample_idx)
        t_display = t_end - t_start
        
        color_matrix = np.zeros((n_display, t_display, 3))
        
        for i, node_idx in enumerate(sample_idx):
            for t_rel, t_abs in enumerate(range(t_start, t_end)):
                color_matrix[i, t_rel] = self.get_node_color(node_idx, t_abs)
        
        return color_matrix


def evaluate_hierarchical_clustering(
    true_super: np.ndarray,
    true_sub: np.ndarray,
    est_super: np.ndarray,
    est_sub: np.ndarray,
) -> Dict:
    """
    Evaluate hierarchical clustering using standard ARI.
    
    Parameters
    ----------
    true_super : np.ndarray, dim n 
        True super-community assignments
    true_sub : np.ndarray, dim T x n 
        True sub-community assignments
    est_super : np.ndarray, dim n 
        Estimated super-community assignments
    est_sub : np.ndarray, dim T x n 
        Estimated sub-community assignments
    
    Returns
    -------
    dict
        Dictionary containing:
        - super_ari: ARI for super-community assignments
        - ari_over_time: List of ARI values for each timepoint
        - mean_ari: Mean ARI of ari_over_time 
    """
    n_nodes = len(true_super)
    T = min(est_sub.shape[0], true_sub.shape[0])
    
    # 1. Super-community ARI
    super_ari = ari(true_super, est_super)
    
    # 2. ARI over time (using hierarchical labels)
    ari_over_time = []
    
    for t in range(T):
        # extract and compare labels in the form super.sub 
        # NB, this does NOT check if super and sub are well aligned, 
        # it ONLY asks if the resulting total label clustering is 
        # correct, modulo renaming.   
        true_labels = [f"{true_super[i]}.{true_sub[t, i]}" 
                      for i in range(n_nodes)]
        est_labels = [f"{est_super[i]}.{est_sub[t, i]}" 
                     for i in range(n_nodes)]
        
        ari_t = ari(true_labels, est_labels)
        ari_over_time.append(ari_t)
    
    # 3. Combined ARI (all node-time pairs)
    true_labels_all = []
    est_labels_all = []
    
    for t in range(T):
        for i in range(n_nodes):
            true_labels_all.append(f"{true_super[i]}.{true_sub[t, i]}")
            est_labels_all.append(f"{est_super[i]}.{est_sub[t, i]}")
    
    combined_ari = ari(true_labels_all, est_labels_all)
    
    return {
        'super_ari': super_ari,
        'ari_over_time': ari_over_time,
        'mean_ari': np.mean(ari_over_time),
        'combined_ari': combined_ari
    }


def evaluate_from_hcs(model, estimation_results) -> Dict:
    """
    Evaluate using HierarchicalCommunityStructure objects.
    
    This is a convenience wrapper that extracts the necessary arrays
    and calls evaluate_hierarchical_clustering.
    """
    
    # Convert to HCS format if needed
    if hasattr(model, 'hierarchical_generator'):
        true_hcs = HierarchicalCommunityStructure.from_model(model)
    else:
        true_hcs = model
        
    est_hcs = HierarchicalCommunityStructure.from_results(estimation_results)

    return evaluate_hierarchical_clustering(
        true_super=true_hcs.super_communities,
        true_sub=true_hcs.sub_communities,
        est_super=est_hcs.super_communities,
        est_sub=est_hcs.sub_communities
    )
