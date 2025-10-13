"""
train.py

Author: Usman Siddiqi
Date: 2025-10-12

This is the main script for training the reinforcement learning agent.
It initializes the PyBullet environment and the agent, then runs the
training loop for a specified number of timesteps. It also handles
saving the trained model.
"""

import os
import time
from datetime import datetime
import argparse

from environment import RobotEnv
from agent import create_agent, create_callbacks, save_agent, get_device_info
from config import TOTAL_TIMESTEPS, MODEL_SAVE_PATH


def train(total_timesteps=TOTAL_TIMESTEPS, 
          save_path=MODEL_SAVE_PATH,
          render_training=True,
          model_name=None):
    """
    Main training function for the robot walking agent.
    
    Args:
        total_timesteps (int): Total number of training timesteps
        save_path (str): Directory to save the trained model
        render_training (bool): Whether to render during training
        model_name (str): Custom name for the saved model
    """
    
    print("=== Robot Walking Training ===")
    print(f"Training timesteps: {total_timesteps:,}")
    print(f"Save path: {save_path}")
    print(f"Render training: {render_training}")
    
    # Show device information
    get_device_info()
    
    # Create the environment
    print("\nCreating environment...")
    env = RobotEnv(render_mode=render_training)
    print("Environment created successfully!")
    
    # Create the agent
    print("\nCreating PPO agent...")
    agent = create_agent(env, verbose=1)
    print("Agent created successfully!")
    
    # Create callbacks for monitoring and saving
    print("\nSetting up training callbacks...")
    callbacks = create_callbacks(env, save_path)
    print("Callbacks configured!")
    
    # Start training
    print(f"\n{'='*50}")
    print("STARTING TRAINING")
    print(f"{'='*50}")
    
    start_time = time.time()
    training_start_datetime = datetime.now()
    
    try:
        # Train the agent
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,        # Log every 10 updates
            reset_num_timesteps=True,
            tb_log_name="robot_walker_training"
        )
        
        training_end_time = time.time()
        training_duration = training_end_time - start_time
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"Training duration: {training_duration/3600:.2f} hours")
        print(f"Started: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n{'='*50}")
        print("TRAINING INTERRUPTED BY USER")
        print(f"{'='*50}")
        training_duration = time.time() - start_time
        print(f"Training duration before interruption: {training_duration/60:.2f} minutes")
        
    except Exception as e:
        print(f"\n{'='*50}")
        print("TRAINING FAILED!")
        print(f"{'='*50}")
        print(f"Error: {str(e)}")
        raise
    
    # Save the final model
    print("\nSaving final model...")
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"robot_walker_final_{timestamp}"
    
    final_model_path = os.path.join(save_path, f"{model_name}.zip")
    save_agent(agent, final_model_path)
    
    # Clean up
    print("\nCleaning up...")
    env.close()
    
    print("\n" + "="*50)
    print("TRAINING SESSION COMPLETED!")
    print("="*50)
    print(f"Final model saved to: {final_model_path}")
    print("You can now use this model for evaluation or further training.")
    print("\nTo run the trained model, use:")
    print(f"python main.py --run --model_path {final_model_path}")


def resume_training(model_path, additional_timesteps, save_path=MODEL_SAVE_PATH):
    """
    Resume training from a previously saved model.
    
    Args:
        model_path (str): Path to the saved model to resume from
        additional_timesteps (int): Additional timesteps to train
        save_path (str): Directory to save the continued model
    """
    
    print("=== Resuming Training ===")
    print(f"Loading model from: {model_path}")
    print(f"Additional timesteps: {additional_timesteps:,}")
    
    # Create environment
    env = RobotEnv(render_mode=False)
    
    # Load the existing model
    from agent import load_trained_agent
    agent = load_trained_agent(model_path, env)
    
    # Continue training
    print("Resuming training...")
    start_time = time.time()
    
    # Create callbacks
    callbacks = create_callbacks(env, save_path)
    
    agent.learn(
        total_timesteps=additional_timesteps,
        callback=callbacks,
        log_interval=10,
        reset_num_timesteps=False,  # Don't reset timestep counter
        tb_log_name="robot_walker_continued"
    )
    
    training_duration = time.time() - start_time
    print(f"Additional training completed in {training_duration/60:.2f} minutes")
    
    # Save the updated model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    continued_model_path = os.path.join(save_path, f"robot_walker_continued_{timestamp}.zip")
    save_agent(agent, continued_model_path)
    
    env.close()
    print(f"Continued model saved to: {continued_model_path}")


def quick_test_training():
    """
    Quick test training with fewer timesteps for debugging and validation.
    """
    
    print("=== Quick Test Training ===")
    print("Running a short training session for testing...")
    
    # Use fewer timesteps for quick testing
    test_timesteps = 10000
    
    # Create environment with rendering for visual feedback
    env = RobotEnv(render_mode=True)
    
    # Create agent
    agent = create_agent(env, verbose=1)
    
    print(f"Starting quick test with {test_timesteps} timesteps...")
    
    # Train for a short time
    agent.learn(
        total_timesteps=test_timesteps,
        log_interval=1,
        tb_log_name="robot_walker_test"
    )
    
    print("Quick test completed!")
    
    # Test the agent for a few episodes
    print("Testing trained agent...")
    obs, _ = env.reset()
    
    for episode in range(3):
        episode_reward = 0
        obs, _ = env.reset()
        
        for step in range(100):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} reward: {episode_reward:.2f}")
    
    env.close()
    print("Quick test session completed!")


def main():
    """
    Main function with command-line argument parsing.
    """
    
    parser = argparse.ArgumentParser(description="Train a robot to walk using reinforcement learning")
    
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                       help=f"Number of training timesteps (default: {TOTAL_TIMESTEPS:,})")
    
    parser.add_argument("--save_path", type=str, default=MODEL_SAVE_PATH,
                       help=f"Directory to save models (default: {MODEL_SAVE_PATH})")
    
    parser.add_argument("--render", action="store_true",
                       help="Render the environment during training (slower)")
    
    parser.add_argument("--model_name", type=str, default=None,
                       help="Custom name for the saved model")
    
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to model to resume training from")
    
    parser.add_argument("--additional_timesteps", type=int, default=100000,
                       help="Additional timesteps when resuming training")
    
    parser.add_argument("--quick_test", action="store_true",
                       help="Run a quick test training session")
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.quick_test:
        quick_test_training()
    elif args.resume:
        resume_training(args.resume, args.additional_timesteps, args.save_path)
    else:
        train(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            render_training=args.render,
            model_name=args.model_name
        )


if __name__ == "__main__":
    main()