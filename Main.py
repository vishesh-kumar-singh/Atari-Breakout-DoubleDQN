from training_function import train_agent
from Visualisation_and_plotting import plot_training_results


if __name__ == "__main__":
    """Main execution block for the assignment"""
    # Display assignment title and information.
    print("Assignment: Using Double Q-Learning along with PER to play Atari Breakout")
    print("Team: Runtime Terrors")
    

    # Execute the training process and collect results.
    agent, episode_rewards, mean_scores, losses = train_agent()

    # Generate and display comprehensive visualizations
    plot_training_results(episode_rewards, mean_scores, losses)

    # Print completion message and file references.
    print('''Training is Completed.
          The model weights are being saved as model_weights.pth.
          And the plots of the results had been saved as png files.''')
    
    # Saving the trained model
    agent.save_model()
    print("Model Weights are saved")