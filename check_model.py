import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torcs_env import TORCSEnv

def check_model_files():
    # Check if model files exist
    model_path = "logs/final_model.zip"
    vec_normalize_path = "logs/vec_normalize.pkl"
    
    print("\nChecking model files:")
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"VecNormalize file exists: {os.path.exists(vec_normalize_path)}")
    
    if not os.path.exists(model_path):
        print("\nERROR: Model file not found!")
        print("Please make sure you have trained the model and it's saved in logs/final_model")
        return False
        
    if not os.path.exists(vec_normalize_path):
        print("\nERROR: VecNormalize file not found!")
        print("Please make sure you have the normalization stats in logs/vec_normalize.pkl")
        return False
    
    print("\nAttempting to load model and environment...")
    try:
        # Create environment
        def make_env():
            env = TORCSEnv()
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        
        # Load model
        model = SAC.load(model_path)
        
        # Test prediction
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        print("\nModel loaded successfully!")
        print(f"Action space shape: {action.shape}")
        print(f"Sample action: {action[0]}")
        print(f"Observation space shape: {obs.shape}")
        print(f"Sample observation: {obs[0][:5]}")  # First 5 values
        
        return True
        
    except Exception as e:
        print(f"\nERROR loading model: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting model verification...")
    success = check_model_files()
    if success:
        print("\nAll checks passed! The model should be ready to use.")
    else:
        print("\nSome checks failed. Please fix the issues before running the game.") 