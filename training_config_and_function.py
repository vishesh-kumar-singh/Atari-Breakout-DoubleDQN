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
    'logging_interval': 25,  # Logging interval
}


import numpy as np
from collections import deque
from Agent_Implementation import AdvancedDQNAgent
from Atari_Environment import AtariWrapper
from tqdm import tqdm

def train_agent():
    """Main training loop with comprehensive monitoring"""
    
    # Initialize the Atari environment wrapper and DQN agent.
    env = AtariWrapper(env_name="BreakoutNoFrameskip-v4", frame_skip=4)
    state=env.reset()
    state_shape = state.shape
    action_size = env.action_space.n
    agent = AdvancedDQNAgent(state_shape, action_size, config=config)

    # Set up tracking variables for metrics and timing.
    episode_rewards = []
    mean_scores = []
    losses = []
    scores_window = deque(maxlen=50)
    total_steps = 0

    # Implement the main episode loop with proper environment interaction.
    for i in range(config['max_episodes']//config['logging_interval']):
        print(f"Training for episode {i*config['logging_interval']+1} to {i*config['logging_interval']+config['logging_interval']}")
        for episode in tqdm(range(config['logging_interval'])):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_loss = []

            while not done:
                # Agent selects action using epsilon-greedy strategy.
                action = agent.select_action(state)

                # Environment takes a step and returns the next state, reward, done flag, truncated flag, and info.
                next_state, reward, done, truncated, info = env.step(action)

                # Handle experience storage, agent updates, and target network synchronization.
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                total_steps += 1

                loss = agent.update()
                if loss is not None:
                    episode_loss.append(loss)

                # Synchronize target network weights at regular intervals.
                if total_steps % agent.update_target_freq == 0:
                    agent.update_target_network()

            # Monitor training progress with comprehensive logging and statistics.
            scores_window.append(episode_reward)
            mean_score = np.mean(scores_window)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            episode_rewards.append(episode_reward)
            mean_scores.append(mean_score)
            losses.append(avg_loss)


        print(f"Episode {(i+1)*config['logging_interval']} | Reward: {episode_reward:.2f} | "
              f"Mean Score: {mean_score:.2f} | Loss: {avg_loss:.4f}")

        # Implement early stopping when target performance is achieved.
        if mean_score >= config['target_score']:
            print(f"\nEnvironment solved in {episode} episodes! Average score: {mean_score:.2f}")
    
    return agent,episode_rewards, mean_scores, losses