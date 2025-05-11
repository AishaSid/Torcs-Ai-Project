import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from torcs_env import TORCSEnv
import argparse

def make_env():
    """Create and wrap the environment."""
    env = TORCSEnv()
    # Monitor needs to be wrapped first
    env = Monitor(env)
    return env

def train(args):
    # Create and wrap the environment
    env = DummyVecEnv([make_env])
    # VecNormalize needs to be wrapped after DummyVecEnv
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create log directory
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create callbacks
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=log_dir,
        name_prefix="torcs_model"
    )
    
    # Initialize the agent with more stable hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                qf=[256, 256]
            )
        ),
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Train the agent
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # Save the final model and environment
        model.save(os.path.join(log_dir, "final_model"))
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        # Save the model even if training fails
        model.save(os.path.join(log_dir, "failed_model"))
        env.save(os.path.join(log_dir, "failed_vec_normalize.pkl"))
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TORCS agent with SAC')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                      help='Total number of timesteps to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=100000,
                      help='Size of the replay buffer')
    parser.add_argument('--learning_starts', type=int, default=1000,
                      help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Minibatch size for each gradient update')
    parser.add_argument('--tau', type=float, default=0.005,
                      help='Target network update rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--train_freq', type=int, default=1,
                      help='Update the model every train_freq steps')
    parser.add_argument('--gradient_steps', type=int, default=1,
                      help='How many gradient steps to do after each rollout')
    parser.add_argument('--ent_coef', type=str, default='auto',
                      help='Entropy regularization coefficient')
    
    args = parser.parse_args()
    train(args) 