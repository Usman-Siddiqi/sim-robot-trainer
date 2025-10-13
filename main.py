"""
main.py

Author: Usman Siddiqi
Date: 2025-10-12

This script serves as the main entry point for the sim-robot-trainer application.
It handles command-line arguments to allow the user to either train a new
model or run and visualize a pre-trained model.
"""

import os
import argparse
import time

from environment import RobotEnv
from agent import load_trained_agent, create_agent
from train import train, quick_test_training
from config import MODEL_SAVE_PATH


def run_trained_model(model_path, num_episodes=5, render=True):
    """
    Run a trained model and visualize its performance.
    
    Args:
        model_path (str): Path to the trained model file
        num_episodes (int): Number of episodes to run
        render (bool): Whether to render the environment
    """
    
    print("=== Running Trained Model ===")
    print(f"Model path: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Create environment
    env = RobotEnv(render_mode=render)
    
    # Load the trained model
    try:
        agent = load_trained_agent(model_path, env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        env.close()
        return
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n{'='*50}")
    print("RUNNING EPISODES")
    print(f"{'='*50}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while True:
            # Get action from trained agent
            action, _ = agent.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                time.sleep(0.01)  # Small delay for visualization
            
            # Print periodic updates
            if episode_length % 100 == 0:
                print(f"  Step {episode_length}, Reward: {episode_reward:.2f}, Height: {info.get('robot_height', 'N/A'):.3f}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Episode {episode + 1} completed:")
        print(f"    Total reward: {episode_reward:.2f}")
        print(f"    Episode length: {episode_length} steps")
        print(f"    Termination reason: {'Fell' if terminated else 'Max steps reached'}")
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("EPISODE SUMMARY")
    print(f"{'='*50}")
    print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Worst reward: {min(episode_rewards):.2f}")
    print(f"Average episode length: {sum(episode_lengths)/len(episode_lengths):.1f} steps")
    print(f"Longest episode: {max(episode_lengths)} steps")
    
    # Clean up
    env.close()
    print("\nModel evaluation completed!")


def list_available_models(models_dir=MODEL_SAVE_PATH):
    """
    List all available trained models in the models directory.
    
    Args:
        models_dir (str): Directory containing saved models
    """
    
    print(f"=== Available Models in {models_dir} ===")
    
    if not os.path.exists(models_dir):
        print("Models directory does not exist.")
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    
    if not model_files:
        print("No trained models found.")
        return []
    
    print(f"Found {len(model_files)} model(s):")
    for i, model_file in enumerate(model_files, 1):
        model_path = os.path.join(models_dir, model_file)
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        mod_time = time.ctime(os.path.getmtime(model_path))
        print(f"  {i}. {model_file}")
        print(f"     Size: {file_size:.1f} MB")
        print(f"     Modified: {mod_time}")
    
    return model_files


def interactive_model_selection(models_dir=MODEL_SAVE_PATH):
    """
    Interactive selection of a model to run.
    
    Args:
        models_dir (str): Directory containing saved models
        
    Returns:
        str: Path to selected model file
    """
    
    model_files = list_available_models(models_dir)
    
    if not model_files:
        return None
    
    while True:
        try:
            choice = input(f"\nSelect a model (1-{len(model_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            model_index = int(choice) - 1
            if 0 <= model_index < len(model_files):
                return os.path.join(models_dir, model_files[model_index])
            else:
                print(f"Please enter a number between 1 and {len(model_files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def main():
    """
    Main function with command-line interface.
    """
    
    parser = argparse.ArgumentParser(
        description="Sim Robot Trainer - Train or run a walking robot using reinforcement learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                          # Start training a new model
  python main.py --run --model_path models/robot_walker_final_20231012_143000.zip
  python main.py --list                          # List available models
  python main.py --interactive                   # Interactive model selection
  python main.py --quick_test                    # Quick test training
        """
    )
    
    # Action selection
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--train", action="store_true",
                             help="Train a new model")
    action_group.add_argument("--run", action="store_true",
                             help="Run a trained model")
    action_group.add_argument("--list", action="store_true",
                             help="List available trained models")
    action_group.add_argument("--interactive", action="store_true",
                             help="Interactive model selection and running")
    action_group.add_argument("--quick_test", action="store_true",
                             help="Run a quick test training session")
    
    # Training options
    training_group = parser.add_argument_group("Training Options")
    training_group.add_argument("--timesteps", type=int, default=1000000,
                               help="Number of training timesteps")
    training_group.add_argument("--render_training", action="store_true",
                               help="Render environment during training (slower)")
    
    # Running options
    running_group = parser.add_argument_group("Running Options")
    running_group.add_argument("--model_path", type=str,
                              help="Path to the trained model file")
    running_group.add_argument("--episodes", type=int, default=5,
                              help="Number of episodes to run")
    running_group.add_argument("--no_render", action="store_true",
                              help="Don't render the environment when running")
    
    # General options
    parser.add_argument("--models_dir", type=str, default=MODEL_SAVE_PATH,
                       help="Directory containing saved models")
    
    args = parser.parse_args()
    
    # Execute the requested action
    if args.train:
        print("Starting training session...")
        train(
            total_timesteps=args.timesteps,
            render_training=args.render_training
        )
        
    elif args.run:
        if not args.model_path:
            print("Error: --model_path is required when using --run")
            return
        
        run_trained_model(
            model_path=args.model_path,
            num_episodes=args.episodes,
            render=not args.no_render
        )
        
    elif args.list:
        list_available_models(args.models_dir)
        
    elif args.interactive:
        print("=== Interactive Model Runner ===")
        model_path = interactive_model_selection(args.models_dir)
        
        if model_path:
            print(f"\nSelected model: {os.path.basename(model_path)}")
            
            # Ask for number of episodes
            while True:
                try:
                    episodes = input(f"Number of episodes to run (default: {args.episodes}): ").strip()
                    if not episodes:
                        episodes = args.episodes
                    else:
                        episodes = int(episodes)
                    break
                except ValueError:
                    print("Please enter a valid number")
            
            run_trained_model(
                model_path=model_path,
                num_episodes=episodes,
                render=not args.no_render
            )
        else:
            print("No model selected. Exiting.")
            
    elif args.quick_test:
        quick_test_training()


if __name__ == "__main__":
    main()