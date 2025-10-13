"""
agent.py

Author: Usman Siddiqi
Date: 2025-10-12

This file defines the reinforcement learning agent. It specifies the
neural network architecture and the RL algorithm (e.g., PPO) to be used.
This script can be used to create, load, and manage the agent model.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

from config import LEARNING_RATE, GAMMA, MODEL_SAVE_PATH


def create_agent(env, verbose=1):
    """
    Create and configure a PPO agent for the robot walking task.
    
    Args:
        env: The robot environment (RobotEnv instance)
        verbose (int): Verbosity level for training output
        
    Returns:
        PPO: Configured PPO agent ready for training
    """
    
    # Create the PPO agent with MlpPolicy
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        n_steps=2048,           # Number of steps to run for each environment per update
        batch_size=64,          # Minibatch size
        n_epochs=10,           # Number of epoch when optimizing the surrogate loss
        gae_lambda=0.95,       # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,        # Clipping parameter for PPO
        ent_coef=0.01,         # Entropy coefficient for exploration
        vf_coef=0.5,           # Value function coefficient
        max_grad_norm=0.5,     # Maximum value for the gradient clipping
        use_sde=False,         # Whether to use State Dependent Exploration
        sde_sample_freq=-1,    # Sample a new noise matrix every n steps
        target_kl=None,        # Target KL divergence (disabled)
        tensorboard_log="./logs/tensorboard/",  # TensorBoard log directory
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # Policy and value network architecture
            activation_fn=torch.nn.ReLU,                   # Activation function
        ),
        device="cpu",         # Use CPU for optimal MlpPolicy performance
        verbose=verbose
    )
    
    return model


def create_callbacks(env, model_save_path=MODEL_SAVE_PATH):
    """
    Create training callbacks for monitoring and saving the model.
    
    Args:
        env: The robot environment
        model_save_path (str): Directory to save model checkpoints
        
    Returns:
        list: List of callbacks for training
    """
    
    # Ensure the save directory exists
    os.makedirs(model_save_path, exist_ok=True)
    
    # Checkpoint callback - saves model every n steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k steps
        save_path=model_save_path,
        name_prefix="robot_walker_checkpoint"
    )
    
    # Create a separate evaluation environment (wrapped in Monitor for logging)
    # We try to instantiate the same env class with render disabled for faster eval.
    try:
        base_env = getattr(env, "unwrapped", env)
        env_class = base_env.__class__
        eval_env = DummyVecEnv([lambda: Monitor(env_class(render_mode=False))])
    except Exception:
        # Fallback: wrap the provided env (less ideal as it shares state)
        eval_env = DummyVecEnv([lambda: Monitor(env)])
    
    # Evaluation callback - evaluates model performance periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path=os.path.join(model_save_path, "eval_logs"),
        eval_freq=25000,  # Evaluate every 25k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    return [checkpoint_callback, eval_callback]


def load_trained_agent(model_path, env):
    """
    Load a pre-trained PPO agent from a saved model file.
    
    Args:
        model_path (str): Path to the saved model file
        env: The robot environment
        
    Returns:
        PPO: Loaded PPO agent
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading trained model from: {model_path}")
    model = PPO.load(model_path, env=env, device="cpu")  # Force CPU device on load

    return model


def save_agent(model, save_path):
    """
    Save the trained agent to a file.
    
    Args:
        model (PPO): The trained PPO model
        save_path (str): Path where to save the model
    """
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Saving model to: {save_path}")
    model.save(save_path)
    print("Model saved successfully!")


def create_vectorized_env(env_class, n_envs=1, **env_kwargs):
    """
    Create a vectorized environment for parallel training.
    
    Args:
        env_class: The environment class (e.g., RobotEnv)
        n_envs (int): Number of parallel environments
        **env_kwargs: Additional arguments for environment creation
        
    Returns:
        DummyVecEnv: Vectorized environment
    """
    
    def make_env():
        def _init():
            env = env_class(**env_kwargs)
            env = Monitor(env)  # Wrap with Monitor for logging
            return env
        return _init
    
    # Create multiple environments
    env_fns = [make_env() for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    return vec_env


def get_device_info():
    """
    Print information about available computing devices.
    """
    
    print("=== Device Information ===")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA available: No")
        print("Using CPU for training")
    
    print("========================")


# Example usage and testing
if __name__ == "__main__":
    from environment import RobotEnv
    
    print("Testing agent creation...")
    
    # Show device information
    get_device_info()
    
    # Create environment
    env = RobotEnv(render_mode=False)
    
    # Create agent
    agent = create_agent(env, verbose=1)
    
    print(f"Agent created successfully!")
    print(f"Policy: {agent.policy}")
    print(f"Environment: {agent.env}")
    print(f"Learning rate: {agent.learning_rate}")
    print(f"Gamma: {agent.gamma}")
    
    # Test agent prediction
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    print(f"Sample action: {action}")
    
    env.close()
    print("Agent test completed!")