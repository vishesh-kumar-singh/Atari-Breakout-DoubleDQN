config = {
    'learning_rate': 2e-4,  # Learning rate for the optimizer
    'gamma': 0.99,  # Discount factor for future rewards
    'buffer_size': 10000,  # Maximum size of the replay buffer
    'batch_size': 16,  # Number of samples to draw from the buffer for each training step
    'target_update_freq': 500,  # Frequency (in training steps) to update the target network
    'initial_replay_size': 5000,  # Minimum number of experiences in the buffer before training starts
    'alpha': 0.6,  # PER prioritization
    'beta_start': 0.4,  # PER importance sampling
    'max_episodes': 2000,  # Maximum number of episodes to run
    'target_score': 30.0,  # Mean score over 50 episodes
    'logging_interval': 25,  # Logging interval
}