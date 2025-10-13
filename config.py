"""
config.py

Author: Usman Siddiqi
Date: 2025-10-12

This file contains the configuration settings and hyperparameters for the
sim-robot-trainer project. Centralizing the configuration makes it easy
to tune the model and simulation without modifying the core logic.
"""

# Reinforcement Learning Hyperparameters
LEARNING_RATE = 3e-4  # Learning rate for the PPO algorithm
GAMMA = 0.99          # Discount factor for future rewards
TOTAL_TIMESTEPS = 1000000  # Total number of timesteps for training

# Model and File Paths
MODEL_SAVE_PATH = "models/"  # Directory to save trained models

# Simulation Parameters
SIMULATION_TIMESTEP = 1/240  # PyBullet simulation timestep (240 Hz)
MAX_EPISODE_STEPS = 1000     # Maximum steps per episode
RENDER_MODE = True           # Set to True for visualization during training

# Robot Physical Parameters
ROBOT_START_HEIGHT = 1.0     # Initial height of robot torso
FALL_THRESHOLD = 0.3         # Height below which robot is considered fallen
FORWARD_REWARD_SCALE = 10.0  # Scaling factor for forward movement reward
FALL_PENALTY = -100.0        # Penalty for falling over

# Action and Observation Space Parameters
NUM_JOINTS = 4               # Number of controllable joints (2 hips, 2 knees)
MAX_TORQUE = 50.0           # Maximum torque that can be applied to joints