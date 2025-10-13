"""
config.py

Author: Usman Siddiqi
Date: 2025-10-12

This file contains the configuration settings and hyperparameters for the
sim-robot-trainer project. Centralizing the configuration makes it easy
to tune the model and simulation without modifying the core logic.
"""

# Reinforcement Learning Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
TOTAL_TIMESTEPS = 5000000  # Reduced slightly, focus on quality

# Model and File Paths
MODEL_SAVE_PATH = "models/"  # Directory to save trained models

# Simulation Parameters
SIMULATION_TIMESTEP = 1/240  # PyBullet simulation timestep (240 Hz)
MAX_EPISODE_STEPS = 1500     # ~6 seconds at 240Hz
RENDER_MODE = False

# Robot Physical Parameters
ROBOT_START_HEIGHT = 0.83  # Realistic standing height
FALL_THRESHOLD = 0.4       # 50% of standing height
FORWARD_REWARD_SCALE = 10.0    # Reduced 10x for balance
FALL_PENALTY = -10.0           # Reduced for exploration
STABILITY_PENALTY_SCALE = 2.0  # New parameter
UPRIGHT_REWARD = 0.5           # Reduced to balance

# Action and Observation Space Parameters
NUM_JOINTS = 4
HIP_MAX_TORQUE = 120.0    # Hips need more torque
KNEE_MAX_TORQUE = 60.0    # Knees need less
MAX_TORQUE = 120.0        # Keep for backwards compatibility