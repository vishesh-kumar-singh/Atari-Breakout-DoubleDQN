config = {
    'learning_rate': 2e-4,  # Learning rate for the optimizer
    'gamma': 0.99,  # Discount factor for future rewards
    'buffer_size': 20000,  # Maximum size of the replay buffer
    'batch_size': 16,  # Number of samples to draw from the buffer for each training step
    'target_update_freq': 500,  # Frequency (in training steps) to update the target network
    'initial_replay_size': 5000,  # Minimum number of experiences in the buffer before training starts
    'alpha': 0.6,  # PER prioritization
    'beta_start': 0.4,  # PER importance sampling
    'max_episodes': 1000,  # Maximum number of episodes to run
    'target_score': 12.0,  # Mean score over 50 episodes
}


def train_agent():
    """Main training loop with comprehensive monitoring"""
    # Initialize the Atari environment wrapper and DQN agent.
    env = AtariWrapper(env_name="BreakoutNoFrameskip-v4", frame_skip=4)
    agent = DQNAgent(config)
    buffer = PrioritizedReplayBuffer(
        capacity=config['buffer_size'],
        alpha=config['alpha'],
        beta_start=config['beta_start']
    )

    # Set up tracking variables for metrics and timing.
    episode_rewards = []
    mean_scores = []
    losses = []
    scores_window = deque(maxlen=50)
    total_steps = 0
    beta = config['beta_start']
    beta_increment = (1.0 - config['beta_start']) / 100000  

    # Implement the main episode loop with proper environment interaction.
    for episode in range(1, config['max_episodes'] + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_loss = []

        while not done:
            # Select action using agent's policy
            action = agent.select_action(state)

            # Interact with the environment
            next_state, reward, done, _ = env.step(action)

            # Handle experience storage, agent updates, and target network synchronization.
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(buffer) > config['initial_replay_size']:
                beta = min(1.0, beta + beta_increment)
                experiences = buffer.sample(config['batch_size'], beta)
                loss = agent.learn(experiences)
                episode_loss.append(loss)

                if total_steps % config['target_update_freq'] == 0:
                    agent.update_target_network()

        # Monitor training progress with comprehensive logging and statistics.
        episode_rewards.append(episode_reward)
        scores_window.append(episode_reward)
        mean_score = np.mean(scores_window)
        mean_scores.append(mean_score)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Mean Score: {mean_score:.2f} | Loss: {avg_loss:.4f}")

        # Implement early stopping when target performance is achieved.
        if mean_score >= config['target_score']:
            print(f"\nEnvironment solved in {episode} episodes! Average Score: {mean_score:.2f}")
            break
