import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import to_rgba
from typing import List, Dict, Optional, Tuple, Union

# current version of the hierarchical colors and structure class 
def generate_hierarchical_colors(n_super=4, n_sub_per_super=4):
    """
    Generate hierarchical colors with 5 distinct hues per super-community.
    """
    color_families = {
        0: {  # Blue family
            'base': (0.0, 0.45, 0.70),      # Medium blue (super-community color)
            0: (0.0, 0.30, 0.55),           # Dark blue
            1: (0.35, 0.70, 0.90),          # Sky blue  
            2: (0.0, 0.60, 0.85),           # Bright blue
            3: (0.85, 0.92, 0.98),          # Very light blue (almost white)
            4: (0.0, 0.15, 0.35),           # Very dark blue
            5: (0.0, 0.10, 0.20),           # Near-black blue
        },
        1: {  # Orange/Red family
            'base': (0.90, 0.60, 0.0),      # Orange (super-community color)
            0: (0.80, 0.40, 0.0),           # Dark orange
            1: (0.95, 0.75, 0.30),          # Yellow-orange
            2: (0.85, 0.35, 0.25),          # Red-orange
            3: (0.98, 0.92, 0.85),          # Very light peach (almost white)
            4: (0.50, 0.20, 0.0),           # Very dark orange
            5: (0.30, 0.10, 0.0),           # Near-black orange
        },
        2: {  # Green family
            'base': (0.0, 0.60, 0.50),      # Teal-green (super-community color)
            0: (0.0, 0.40, 0.30),           # Dark green
            1: (0.35, 0.70, 0.45),          # Light green
            2: (0.0, 0.75, 0.65),           # Aqua green
            3: (0.85, 0.98, 0.92),          # Very pale green (almost white)
            4: (0.0, 0.25, 0.15),           # Very dark green
            5: (0.0, 0.15, 0.10),           # Near-black green
        },
        3: {  # Purple family
            'base': (0.80, 0.40, 0.70),     # Purple-pink (super-community color)
            0: (0.60, 0.20, 0.50),          # Dark purple
            1: (0.90, 0.60, 0.80),          # Light pink
            2: (0.70, 0.35, 0.85),          # Blue-purple
            3: (0.98, 0.90, 0.95),          # Very pale pink (almost white)
            4: (0.40, 0.10, 0.30),          # Very dark purple
            5: (0.25, 0.05, 0.20),          # Near-black purple
        },
        4: {  # Grey/Neutral family
            'base': (0.50, 0.50, 0.50),      # Medium grey
            0: (0.25, 0.25, 0.25),           # Dark grey
            1: (0.65, 0.65, 0.65),           # Light grey
            2: (0.40, 0.40, 0.40),           # Charcoal grey
            3: (0.92, 0.92, 0.92),           # Very light grey (almost white)
            4: (0.15, 0.15, 0.15),           # Very dark grey
            5: (0.05, 0.05, 0.05),           # Near-black grey
        },
        5: {  # Brown/Warm neutral family
            'base': (0.55, 0.40, 0.25),      # Medium brown
            0: (0.35, 0.25, 0.15),           # Dark brown
            1: (0.70, 0.55, 0.40),           # Tan/light brown
            2: (0.60, 0.35, 0.20),           # Reddish brown
            3: (0.94, 0.90, 0.86),           # Very pale beige (almost white)
            4: (0.20, 0.15, 0.10),           # Very dark brown
            5: (0.10, 0.08, 0.05),           # Near-black brown
        },
        6: {  # Yellow/Gold family
            'base': (0.90, 0.75, 0.10),      # Gold yellow
            0: (0.70, 0.55, 0.0),            # Dark gold
            1: (0.95, 0.85, 0.40),           # Light yellow
            2: (0.85, 0.70, 0.0),            # Amber yellow
            3: (0.98, 0.96, 0.88),           # Very pale yellow (almost white)
            4: (0.45, 0.35, 0.0),            # Very dark gold
            5: (0.25, 0.20, 0.0),            # Near-black yellow
        }
    }
    colors = {}
    base_colors = {}
    
    for sc in range(n_super):
        family = color_families[sc % len(color_families)]
        base_colors[sc] = family['base']
        
        for sub in range(n_sub_per_super):
            colors[(sc, sub)] = family[sub]
    
    return colors, base_colors, color_families


def compute_tsne_embeddings(X_list: List[np.ndarray], 
                           perplexity: int = 30,
                           random_state: int = 42) -> List[np.ndarray]:
    """
    Compute t-SNE embeddings for a list of high-dimensional embeddings.
    
    Important: t-SNE is fit on ALL data points across all time steps to ensure
    consistent embedding space, then split back into individual time steps.
    
    Parameters:
    -----------
    X_list : List[np.ndarray]
        List of embeddings, one per time step. Each is n_nodes x d.
    perplexity : int
        t-SNE perplexity parameter
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    List[np.ndarray]
        List of 2D embeddings, one per time step. Each is n_nodes x 2.
    """
    # Stack all embeddings
    X_combined = np.vstack(X_list)
    n_nodes = X_list[0].shape[0]
    n_timepoints = len(X_list)
    
    # Fit t-SNE on all data
    tsne = TSNE(n_components=2, 
                random_state=random_state,
                perplexity=min(perplexity, X_combined.shape[0] // 4))
    
    X_2d_combined = tsne.fit_transform(X_combined)
    
    # Split back into time steps
    X_2d_list = [X_2d_combined[t*n_nodes:(t+1)*n_nodes] for t in range(n_timepoints)]
    
    return X_2d_list


def plot_embeddings(X_list: List[np.ndarray],
                   labels_list: List[np.ndarray],
                   time_points: Optional[List[int]] = None,
                   color_by: str = 'community',  # 'community' or 'subgraph'
                   hierarchical_labels: Optional[List[List[Tuple[int, int]]]] = None,
                   title: str = "Network Embeddings Over Time",
                   perplexity: int = 30,
                   random_state: int = 42,
                   figsize: Optional[Tuple[int, int]] = None,
                   save_path: Optional[str] = None):
    """
    Plot network embeddings over time using t-SNE visualization.
    
    Parameters:
    -----------
    X_list : List[np.ndarray]
        List of embeddings, one per time step. Each is n_nodes x d.
    labels_list : List[np.ndarray]
        List of label arrays for coloring. Shape depends on color_by:
        - If 'community': simple community labels
        - If 'subgraph': subgraph labels
    time_points : List[int], optional
        Time points for labeling. If None, uses 0, 1, 2, ...
    color_by : str
        'community' or 'subgraph' for hierarchical models
    hierarchical_labels : List[List[Tuple[int, int]]], optional
        For hierarchical models: list of (subgraph, community) tuples per time
    title : str
        Plot title
    perplexity : int
        t-SNE perplexity parameter
    random_state : int
        Random seed
    figsize : Tuple[int, int], optional
        Figure size. If None, automatically determined
    save_path : str, optional
        Path to save the figure
    """
    n_timepoints = len(X_list)
    if time_points is None:
        time_points = list(range(n_timepoints))
    
    # Compute t-SNE embeddings
    X_2d_list = compute_tsne_embeddings(X_list, perplexity, random_state)
    
    # Set up colors
    if hierarchical_labels is not None and color_by == 'subgraph':
        # Color by subgraph
        all_subgraphs = set()
        for labels in hierarchical_labels:
            all_subgraphs.update(sg for sg, _ in labels)
        n_colors = len(all_subgraphs)
        colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        
    else:
        # Color by community or simple labels
        all_labels = np.unique(np.concatenate(labels_list))
        n_colors = len(all_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
    
    # Create figure
    if figsize is None:
        figsize = (4 * n_timepoints, 4)
    
    fig, axes = plt.subplots(1, n_timepoints, figsize=figsize)
    if n_timepoints == 1:
        axes = [axes]
    
    # Plot each time point
    for idx, t in enumerate(time_points):
        X_2d = X_2d_list[idx]
        ax = axes[idx]
        
        if hierarchical_labels is not None:
            # Hierarchical coloring
            for i, (sg, comm) in enumerate(hierarchical_labels[idx]):
                if color_by == 'subgraph':
                    color_idx = sg
                else:  # color by community within subgraph
                    # Create unique color for each (subgraph, community) pair
                    color_idx = sg * 10 + comm  # Assumes < 10 communities per subgraph
                    color_idx = color_idx % n_colors
                
                ax.scatter(X_2d[i, 0], X_2d[i, 1],
                          c=[to_rgba(colors[color_idx], alpha=0.7)],
                          s=30, edgecolors='none')
        else:
            # Simple coloring
            labels = labels_list[idx]
            for i, label in enumerate(labels):
                color_idx = np.where(all_labels == label)[0][0]
                ax.scatter(X_2d[i, 0], X_2d[i, 1],
                          c=[to_rgba(colors[color_idx], alpha=0.7)],
                          s=30, edgecolors='none')
        
        ax.set_title(f't = {t}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hierarchical_embeddings(model,
                                time_points: List[int],
                                embedding_type: str = 'latent',  # 'latent' or 'ase' or 'uase'
                                color_by: str = 'both',  # 'subgraph', 'community', or 'both'
                                d: int = 10,
                                perplexity: int = 30,
                                random_state: int = 42,
                                save_path: Optional[str] = None):
    """
    Convenience function to plot embeddings from a hierarchical model.
    
    Parameters:
    -----------
    model : HierarchicalDynamicDCSBM
        The hierarchical model
    time_points : List[int]
        Time points to visualize
    embedding_type : str
        Type of embedding:
        - 'latent': Use latent positions X from the model
        - 'ase': Compute ASE from adjacency matrices
        - 'uase': Compute UASE from adjacency matrices
    color_by : str
        - 'subgraph': Color by subgraph only
        - 'community': Color by community (ignoring subgraph)
        - 'both': Create separate plots for each
    d : int
        Dimension for ASE/UASE (ignored for latent)
    perplexity : int
        t-SNE perplexity
    random_state : int
        Random seed
    save_path : str, optional
        Base path for saving (will append _subgraph or _community)
    """
    # Get embeddings based on type
    if embedding_type == 'latent':
        X_list = [model.get_X(t) for t in time_points]
    elif embedding_type == 'ase':
        from dyn_dcsbm import adjacency_spectral_embedding
        X_list = [adjacency_spectral_embedding(model.generate_adjacency_matrix(t), d) 
                  for t in time_points]
    elif embedding_type == 'uase':
        from dyn_dcsbm import unfolded_adjacency_spectral_embedding
        adjacency_matrices = [model.generate_adjacency_matrix(t) for t in time_points]
        X_list = unfolded_adjacency_spectral_embedding(adjacency_matrices, d)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    # Get hierarchical labels
    hierarchical_labels = []
    for t in time_points:
        labels_t = []
        for i in range(model.n):
            sg_id, comm_id, _ = model.hierarchical_generator.get_hierarchical_assignment(t, i)
            labels_t.append((sg_id, comm_id))
        hierarchical_labels.append(labels_t)
    
    # Create appropriate plots
    if color_by == 'both':
        # Plot colored by subgraph
        plot_embeddings(X_list, 
                       labels_list=None,
                       time_points=time_points,
                       color_by='subgraph',
                       hierarchical_labels=hierarchical_labels,
                       title=f"{embedding_type.upper()} Embeddings - Colored by Subgraph",
                       perplexity=perplexity,
                       random_state=random_state,
                       save_path=f"{save_path}_subgraph.png" if save_path else None)
        
        # Plot colored by community
        # Flatten to simple community labels
        community_labels_list = []
        for labels_t in hierarchical_labels:
            # Create unique community IDs across subgraphs
            comm_labels = []
            for sg, comm in labels_t:
                # Offset community IDs by subgraph
                if sg == 0:
                    comm_labels.append(comm)
                elif sg == 1:
                    comm_labels.append(2 + comm)  # Subgraph 0 has 2 communities
                else:  # sg == 2
                    comm_labels.append(5 + comm)  # Subgraph 0 has 2, subgraph 1 has 3
            community_labels_list.append(np.array(comm_labels))
        
        plot_embeddings(X_list,
                       labels_list=community_labels_list,
                       time_points=time_points,
                       color_by='community',
                       title=f"{embedding_type.upper()} Embeddings - Colored by Community",
                       perplexity=perplexity,
                       random_state=random_state,
                       save_path=f"{save_path}_community.png" if save_path else None)
    
    else:
        # Single plot with specified coloring
        if color_by == 'community':
            # Flatten to simple community labels as above
            community_labels_list = []
            for labels_t in hierarchical_labels:
                comm_labels = []
                for sg, comm in labels_t:
                    if sg == 0:
                        comm_labels.append(comm)
                    elif sg == 1:
                        comm_labels.append(2 + comm)
                    else:
                        comm_labels.append(5 + comm)
                community_labels_list.append(np.array(comm_labels))
            
            plot_embeddings(X_list,
                           labels_list=community_labels_list,
                           time_points=time_points,
                           title=f"{embedding_type.upper()} Embeddings - Colored by Community",
                           perplexity=perplexity,
                           random_state=random_state,
                           save_path=save_path)
        else:
            plot_embeddings(X_list,
                           labels_list=None,
                           time_points=time_points,
                           color_by=color_by,
                           hierarchical_labels=hierarchical_labels,
                           title=f"{embedding_type.upper()} Embeddings - Colored by {color_by.capitalize()}",
                           perplexity=perplexity,
                           random_state=random_state,
                           save_path=save_path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional, Dict, Union


def hierarchical_reindex(n: int, 
                        hierarchical_labels: List[Tuple[int, int]]) -> Tuple[np.ndarray, Dict]:
    """
    Create a reindexing for nodes based on hierarchical (subgraph, block) structure.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    hierarchical_labels : List[Tuple[int, int]]
        List of (subgraph_id, block_id) for each node
        
    Returns:
    --------
    reindex : np.ndarray
        Array where reindex[new_position] = original_node_id
    boundaries : dict
        Dictionary with boundary information:
        - 'subgraph': List of positions where subgraph boundaries occur
        - 'block': List of positions where block boundaries occur
        - 'sizes': Dict mapping (subgraph, block) to size
    """
    # Create sorting keys: (subgraph, block, original_index)
    sorting_keys = [(sg, block, i) for i, (sg, block) in enumerate(hierarchical_labels)]
    sorted_keys = sorted(sorting_keys)
    
    # Extract reindexing
    reindex = np.array([x[2] for x in sorted_keys])
    
    # Find boundaries
    subgraph_boundaries = []
    block_boundaries = []
    sizes = {}
    
    last_sg, last_block = None, None
    current_block_start = 0
    
    for pos, (sg, block, _) in enumerate(sorted_keys):
        if last_sg is not None and sg != last_sg:
            # New subgraph
            subgraph_boundaries.append(pos)
            block_boundaries.append(pos)
            # Record size of last block
            sizes[(last_sg, last_block)] = pos - current_block_start
            current_block_start = pos
            
        elif last_block is not None and block != last_block:
            # New block within same subgraph
            block_boundaries.append(pos)
            # Record size of last block
            sizes[(last_sg, last_block)] = pos - current_block_start
            current_block_start = pos
        
        last_sg, last_block = sg, block
    
    # Don't forget the last block
    if last_sg is not None:
        sizes[(last_sg, last_block)] = n - current_block_start
    
    boundaries = {
        'subgraph': subgraph_boundaries,
        'block': block_boundaries,
        'sizes': sizes
    }
    
    return reindex, boundaries


def plot_adjacency_matrix(A: np.ndarray,
                         hierarchical_labels: Optional[List[Tuple[int, int]]] = None,
                         node_order: Optional[np.ndarray] = None,
                         boundaries: Optional[Dict] = None,
                         title: str = "Adjacency Matrix",
                         figsize: Tuple[int, int] = (8, 8),
                         cmap: str = 'binary',
                         show_boundaries: bool = True,
                         boundary_colors: Dict[str, str] = None,
                         boundary_widths: Dict[str, float] = None,
                         save_path: Optional[str] = None):
    """
    Plot adjacency matrix with optional hierarchical ordering and boundaries.
    
    Parameters:
    -----------
    A : np.ndarray
        Adjacency matrix to plot
    hierarchical_labels : List[Tuple[int, int]], optional
        List of (subgraph_id, block_id) for each node. If provided, will reorder.
    node_order : np.ndarray, optional
        Custom node ordering. If None and hierarchical_labels provided, will compute.
    boundaries : dict, optional
        Boundary information. If None and hierarchical_labels provided, will compute.
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap for the matrix
    show_boundaries : bool
        Whether to draw boundary lines
    boundary_colors : dict, optional
        Colors for boundaries. Defaults: {'subgraph': 'red', 'block': 'blue'}
    boundary_widths : dict, optional
        Line widths for boundaries. Defaults: {'subgraph': 2, 'block': 1}
    save_path : str, optional
        Path to save the figure
    """
    # Set defaults
    if boundary_colors is None:
        boundary_colors = {'subgraph': 'red', 'block': 'blue'}
    if boundary_widths is None:
        boundary_widths = {'subgraph': 2, 'block': 1}
    
    # Handle reordering
    if hierarchical_labels is not None and node_order is None:
        node_order, boundaries = hierarchical_reindex(len(A), hierarchical_labels)
    
    # Apply reordering if provided
    if node_order is not None:
        A_reordered = A[np.ix_(node_order, node_order)]
    else:
        A_reordered = A
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot matrix
    im = ax.imshow(A_reordered, cmap=cmap, aspect='equal')
    
    # Add boundaries if requested
    if show_boundaries and boundaries is not None:
        # First, identify subgraph ranges
        subgraph_ranges = []
        current_start = 0
        
        for pos in boundaries['subgraph'] + [len(A)]:
            subgraph_ranges.append((current_start, pos))
            current_start = pos
        
        # Draw block boundaries within each subgraph
        for pos in boundaries['block']:
            if pos not in boundaries['subgraph']:  # Don't double-draw
                # Find which subgraph this boundary belongs to
                for sg_start, sg_end in subgraph_ranges:
                    if sg_start < pos < sg_end:
                        # Draw lines only within this subgraph's range
                        # For horizontal line at y=pos-0.5, x goes from sg_start to sg_end
                        ax.plot([sg_start, sg_end], [pos-0.5, pos-0.5], 
                               color=boundary_colors['block'], 
                               linewidth=boundary_widths['block'], alpha=0.7)
                        # For vertical line at x=pos-0.5, y goes from sg_start to sg_end
                        ax.plot([pos-0.5, pos-0.5], [sg_start, sg_end],
                               color=boundary_colors['block'], 
                               linewidth=boundary_widths['block'], alpha=0.7)
                        break
        
        # Draw subgraph boundaries (limited to matrix size)
        n = len(A)
        for pos in boundaries['subgraph']:
            # Horizontal line from x=0 to x=n
            ax.plot([0, n], [pos-0.5, pos-0.5],
                   color=boundary_colors['subgraph'], 
                   linewidth=boundary_widths['subgraph'], alpha=0.8)
            # Vertical line from y=0 to y=n
            ax.plot([pos-0.5, pos-0.5], [0, n],
                   color=boundary_colors['subgraph'], 
                   linewidth=boundary_widths['subgraph'], alpha=0.8)
    
    # Set axis limits to exactly match the data
    ax.set_xlim(-0.5, len(A)-0.5)
    ax.set_ylim(len(A)-0.5, -0.5)  # Note: y-axis is inverted for images
    
    # Styling
    ax.set_title(title)
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Node Index")
    
   
    # Add legend for boundaries
    if show_boundaries and boundaries is not None:
        from matplotlib.lines import Line2D
        legend_elements = []
        if boundaries['subgraph']:
            legend_elements.append(Line2D([0], [0], color=boundary_colors['subgraph'], 
                                        lw=boundary_widths['subgraph'], label='Subgraph'))
        if boundaries['block']:
            legend_elements.append(Line2D([0], [0], color=boundary_colors['block'], 
                                        lw=boundary_widths['block'], label='Block'))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_adjacency_comparison(A: np.ndarray,
                             true_labels: List[Tuple[int, int]],
                             pred_labels: List[Tuple[int, int]],
                             title: str = "True vs Predicted Structure",
                             figsize: Tuple[int, int] = (16, 8),
                             save_path: Optional[str] = None):
    """
    Plot side-by-side comparison of adjacency matrix with true and predicted orderings.
    
    Parameters:
    -----------
    A : np.ndarray
        Adjacency matrix
    true_labels : List[Tuple[int, int]]
        True (subgraph, block) labels
    pred_labels : List[Tuple[int, int]]
        Predicted (subgraph, block) labels
    title : str
        Overall title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # True ordering
    true_order, true_boundaries = hierarchical_reindex(len(A), true_labels)
    A_true = A[np.ix_(true_order, true_order)]
    
    im1 = ax1.imshow(A_true, cmap='binary', aspect='equal')
    ax1.set_title("True Structure")
    
    # Add true boundaries
    # First identify subgraph ranges
    true_sg_ranges = []
    current_start = 0
    for pos in true_boundaries['subgraph'] + [len(A)]:
        true_sg_ranges.append((current_start, pos))
        current_start = pos
    
    # Draw block boundaries within subgraphs
    for pos in true_boundaries['block']:
        if pos not in true_boundaries['subgraph']:
            for sg_start, sg_end in true_sg_ranges:
                if sg_start < pos < sg_end:
                    ax1.plot([sg_start, sg_end], [pos-0.5, pos-0.5],
                            color='blue', linewidth=1, alpha=0.7)
                    ax1.plot([pos-0.5, pos-0.5], [sg_start, sg_end],
                            color='blue', linewidth=1, alpha=0.7)
                    break
    
    # Draw subgraph boundaries (limited to matrix size)
    n = len(A)
    for pos in true_boundaries['subgraph']:
        ax1.plot([0, n], [pos-0.5, pos-0.5],
                color='red', linewidth=2, alpha=0.8)
        ax1.plot([pos-0.5, pos-0.5], [0, n],
                color='red', linewidth=2, alpha=0.8)
    
    # Set axis limits
    ax1.set_xlim(-0.5, n-0.5)
    ax1.set_ylim(n-0.5, -0.5)
    
    # Predicted ordering
    pred_order, pred_boundaries = hierarchical_reindex(len(A), pred_labels)
    A_pred = A[np.ix_(pred_order, pred_order)]
    
    im2 = ax2.imshow(A_pred, cmap='binary', aspect='equal')
    ax2.set_title("Predicted Structure")
    
    # Add predicted boundaries
    # First identify subgraph ranges
    pred_sg_ranges = []
    current_start = 0
    for pos in pred_boundaries['subgraph'] + [len(A)]:
        pred_sg_ranges.append((current_start, pos))
        current_start = pos
    
    # Draw block boundaries within subgraphs
    for pos in pred_boundaries['block']:
        if pos not in pred_boundaries['subgraph']:
            for sg_start, sg_end in pred_sg_ranges:
                if sg_start < pos < sg_end:
                    ax2.plot([sg_start, sg_end], [pos-0.5, pos-0.5],
                            color='blue', linewidth=1, alpha=0.7)
                    ax2.plot([pos-0.5, pos-0.5], [sg_start, sg_end],
                            color='blue', linewidth=1, alpha=0.7)
                    break
    
    # Draw subgraph boundaries (limited to matrix size)
    n = len(A)
    for pos in pred_boundaries['subgraph']:
        ax2.plot([0, n], [pos-0.5, pos-0.5],
                color='red', linewidth=2, alpha=0.8)
        ax2.plot([pos-0.5, pos-0.5], [0, n],
                color='red', linewidth=2, alpha=0.8)
    
    # Set axis limits
    ax2.set_xlim(-0.5, n-0.5)
    ax2.set_ylim(n-0.5, -0.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Subgraph'),
        Line2D([0], [0], color='blue', lw=1, label='Block')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Overall title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hierarchical_adjacency_matrices(model,
                                        time_points: List[int],
                                        comparison: str = 'single',  # 'single' or 'true_vs_pred'
                                        pred_labels_dict: Optional[Dict[int, List[Tuple[int, int]]]] = None,
                                        title_prefix: Optional[str] = None,
                                        figsize: Optional[Tuple[int, int]] = None,
                                        save_prefix: Optional[str] = None):
    """
    Convenience function to plot adjacency matrices from hierarchical model.
    
    Parameters:
    -----------
    model : HierarchicalDynamicDCSBM
        The hierarchical model
    time_points : List[int]
        Time points to visualize
    comparison : str
        'single': Just show the true structure
        'true_vs_pred': Show true vs predicted (requires pred_labels_dict)
    pred_labels_dict : Dict[int, List[Tuple[int, int]]], optional
        Dictionary mapping time -> predicted labels
    title_prefix : str, optional
        Prefix for titles. If None, uses default titles
    figsize : Tuple[int, int], optional
        Figure size per plot
    save_prefix : str, optional
        Prefix for saved files
    """
    for t in time_points:
        A = model.generate_adjacency_matrix(t)
        
        # Get true labels
        true_labels = []
        for i in range(model.n):
            sg_id, comm_id, _ = model.hierarchical_generator.get_hierarchical_assignment(t, i)
            true_labels.append((sg_id, comm_id))
        
        if comparison == 'single':
            # Single plot with true structure
            if title_prefix:
                title = f"{title_prefix} (t={t})"
            else:
                title = f"Adjacency Matrix at t={t}"
                
            plot_adjacency_matrix(
                A,
                hierarchical_labels=true_labels,
                title=title,
                figsize=figsize or (8, 8),
                save_path=f"{save_prefix}_t{t}.png" if save_prefix else None
            )
        
        elif comparison == 'true_vs_pred' and pred_labels_dict is not None:
            # Comparison plot
            if t in pred_labels_dict:
                if title_prefix:
                    title = f"{title_prefix} (t={t}): True vs Predicted"
                else:
                    title = f"Adjacency Matrix at t={t}: True vs Predicted"
                    
                plot_adjacency_comparison(
                    A,
                    true_labels=true_labels,
                    pred_labels=pred_labels_dict[t],
                    title=title,
                    figsize=figsize or (16, 8),
                    save_path=f"{save_prefix}_comparison_t{t}.png" if save_prefix else None
                )


# Utility function to get block statistics
def analyze_block_structure(A: np.ndarray, 
                           hierarchical_labels: List[Tuple[int, int]]) -> Dict:
    """
    Analyze block structure of adjacency matrix.
    
    Returns dictionary with:
    - Block densities
    - Within/between subgraph edge counts
    - Block sizes
    """
    reindex, boundaries = hierarchical_reindex(len(A), hierarchical_labels)
    A_reordered = A[np.ix_(reindex, reindex)]
    
    stats = {
        'block_densities': {},
        'subgraph_edge_counts': {},
        'block_sizes': boundaries['sizes']
    }
    
    # Calculate block densities
    pos = 0
    for (sg, block), size in boundaries['sizes'].items():
        if size > 0:
            block_adj = A_reordered[pos:pos+size, pos:pos+size]
            density = np.sum(block_adj) / (size * (size - 1)) if size > 1 else 0
            stats['block_densities'][(sg, block)] = density
            pos += size
    
    # Count within/between subgraph edges
    for sg in set(sg for sg, _ in boundaries['sizes'].keys()):
        stats['subgraph_edge_counts'][sg] = {
            'within': 0,
            'between': 0
        }
    
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[i, j] == 1:
                sg_i = hierarchical_labels[i][0]
                sg_j = hierarchical_labels[j][0]
                if sg_i == sg_j:
                    stats['subgraph_edge_counts'][sg_i]['within'] += 1
                else:
                    stats['subgraph_edge_counts'][sg_i]['between'] += 1
                    stats['subgraph_edge_counts'][sg_j]['between'] += 1
    
    return stats


