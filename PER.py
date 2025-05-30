class SumTree:
    def __init__(self, capacity: int):
        """Initialize tree structure for efficient priority sampling"""
        # Setup binary tree arrays and tracking variables
        pass
    
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority changes up the tree"""
        # Update parent nodes recursively to root
        pass
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for given priority value"""
        # Navigate tree based on cumulative priorities
        pass
    
    def total_priority(self) -> float:
        """Return total priority (root value)"""
        return self.tree[0]
    
    def add(self, priority: float, data: object) -> None:
        """Store experience with priority"""
        # Add data to tree and propagate changes
        pass
    
    def update(self, idx: int, priority: float) -> None:
        """Update priority of existing experience"""
        # Update node and propagate change
        pass
    
    def get_leaf(self, s: float) -> Tuple[int, float, object]:
        """Sample leaf based on priority value s"""
        # Find leaf and return index, priority, data
        pass

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        """Initialize prioritized replay buffer"""
        # Set up the SumTree and store prioritization parameters.
        # Initialize beta annealing schedule and numerical stability constants.
        # Set up frame counting for importance sampling weight calculation.
        pass
    
    def _get_beta(self) -> float:
        """Calculate current beta value (anneals to 1.0)"""
        # Implement linear annealing from beta_start to 1.0 over training.
        pass
    
    def push(self, state, action, reward, next_state, done) -> None:
        """Store experience with maximum priority"""
        # Package the experience tuple and assign appropriate priority.
        # Use maximum existing priority for new experiences to ensure sampling.
        pass
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        # Divide priority range into segments for stratified sampling.
        # Calculate importance sampling weights to correct for sampling bias.
        # Return experiences, tree indices, and normalized weights.
        pass
    
    def update_priorities(self, indices, priorities) -> None:
        """Update priorities based on TD errors"""
        # Convert TD errors to priorities using alpha exponent.
        # Add small epsilon for numerical stability.
        # Update tree nodes with new priority values.
        pass
    
    def __len__(self) -> int:
        """Return current buffer size"""
        # Return the number of experiences currently stored.
        pass