import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torcs_env import TORCSEnv
import matplotlib.pyplot as plt

def make_env():
    """Create and wrap the environment."""
    env = TORCSEnv()
    env = Monitor(env)
    return env

def test_agent(model_path, vec_normalize_path, num_episodes=5):
    # Create and wrap the environment
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False  # Disable training mode
    env.norm_reward = False  # Disable reward normalization for evaluation
    
    # Load the trained model
    model = SAC.load(model_path)
    
    # Lists to store episode data
    episode_rewards = []
    episode_lengths = []
    episode_speeds = []
    episode_track_positions = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_speed = []
        episode_track_pos = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += float(reward[0])  # Convert reward to float
            episode_length += 1
            episode_speed.append(float(info[0]['speed']))
            episode_track_pos.append(float(info[0]['track_pos']))
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_speeds.append(np.mean(episode_speed))
        episode_track_positions.append(np.mean(episode_track_pos))
        
        print(f"\nEpisode {episode + 1}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Episode Length: {episode_length}")
        print(f"Average Speed: {np.mean(episode_speed):.2f}")
        print(f"Average Track Position: {np.mean(episode_track_pos):.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot average speeds
    plt.subplot(2, 2, 3)
    plt.plot(episode_speeds)
    plt.title('Average Speeds')
    plt.xlabel('Episode')
    plt.ylabel('Speed')
    
    # Plot average track positions
    plt.subplot(2, 2, 4)
    plt.plot(episode_track_positions)
    plt.title('Average Track Positions')
    plt.xlabel('Episode')
    plt.ylabel('Track Position')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average Speed: {np.mean(episode_speeds):.2f} ± {np.std(episode_speeds):.2f}")
    print(f"Average Track Position: {np.mean(episode_track_positions):.2f} ± {np.std(episode_track_positions):.2f}")

if __name__ == "__main__":
    # Paths to the saved model and normalization
    model_path = "logs/final_model"
    vec_normalize_path = "logs/vec_normalize.pkl"
    
    # Run evaluation
    test_agent(model_path, vec_normalize_path) 