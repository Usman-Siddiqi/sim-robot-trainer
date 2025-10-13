"""
environment.py

Author: Usman Siddiqi
Date: 2025-10-12

This script defines the simulation environment for the robot using PyBullet.
It contains the RobotEnv class, which handles the physics simulation,
robot state, actions, and reward calculation. This class is designed to be
compatible with reinforcement learning libraries like Stable Baselines3.
"""

import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import time

from config import (
    SIMULATION_TIMESTEP, MAX_EPISODE_STEPS, ROBOT_START_HEIGHT,
    FALL_THRESHOLD, FORWARD_REWARD_SCALE, FALL_PENALTY,
    NUM_JOINTS, MAX_TORQUE, RENDER_MODE
)


class RobotEnv(gym.Env):
    """
    Custom environment for training a bipedal robot to walk using PyBullet.
    
    This environment creates a simple bipedal robot programmatically and
    provides the OpenAI Gym interface for reinforcement learning.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, render_mode=RENDER_MODE):
        """
        Initialize the robot environment.
        
        Args:
            render_mode (bool): Whether to render the simulation visually
        """
        super(RobotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.episode_steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        
        # Connect to PyBullet
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up PyBullet environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIMULATION_TIMESTEP)
        p.setRealTimeSimulation(0)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Robot body parts and joints
        self.robot_id = None
        self.joint_indices = []
        self.initial_position = [0, 0, ROBOT_START_HEIGHT]
        
        # Define action space: torques for 4 joints (2 hips, 2 knees)
        self.action_space = spaces.Box(
            low=-MAX_TORQUE,
            high=MAX_TORQUE,
            shape=(NUM_JOINTS,),
            dtype=np.float32
        )
        
        # Define observation space: joint angles, velocities, torso orientation, and position
        # 4 joint angles + 4 joint velocities + 3 torso orientation + 3 torso position + 3 torso linear velocity
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32
        )
        
        # Create the robot
        self._create_robot()
        
        # Previous position for calculating forward movement
        self.prev_position = None
        
    def _create_robot(self):
        """
        Programmatically create a simple bipedal robot using basic PyBullet shapes.
        """
        # Robot dimensions
        torso_height = 0.6
        torso_width = 0.3
        torso_depth = 0.2
        
        upper_leg_length = 0.4
        lower_leg_length = 0.4
        leg_radius = 0.05
        
        # Create torso (main body)
        torso_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[torso_width/2, torso_depth/2, torso_height/2]
        )
        torso_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[torso_width/2, torso_depth/2, torso_height/2],
            rgbaColor=[0.7, 0.7, 0.7, 1]
        )
        
        # Create upper legs
        upper_leg_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=leg_radius,
            height=upper_leg_length
        )
        upper_leg_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=leg_radius,
            length=upper_leg_length,
            rgbaColor=[0.5, 0.5, 0.8, 1]
        )
        
        # Create lower legs
        lower_leg_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=leg_radius,
            height=lower_leg_length
        )
        lower_leg_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=leg_radius,
            length=lower_leg_length,
            rgbaColor=[0.8, 0.5, 0.5, 1]
        )
        
        # Create feet
        foot_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.05, 0.02]
        )
        foot_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.05, 0.02],
            rgbaColor=[0.2, 0.2, 0.2, 1]
        )
        
        # Mass properties
        torso_mass = 5.0
        leg_mass = 1.0
        foot_mass = 0.5
        
        # Link positions relative to parent
        hip_offset = 0.15  # Distance from torso center to hip
        
        # Create multi-body robot
        link_masses = [leg_mass, leg_mass, foot_mass, leg_mass, leg_mass, foot_mass]
        link_collision_shapes = [upper_leg_collision, lower_leg_collision, foot_collision,
                               upper_leg_collision, lower_leg_collision, foot_collision]
        link_visual_shapes = [upper_leg_visual, lower_leg_visual, foot_visual,
                            upper_leg_visual, lower_leg_visual, foot_visual]
        
        # Link positions (relative to parent link frame)
        link_positions = [
            [hip_offset, 0, -torso_height/2],     # Right upper leg
            [0, 0, -upper_leg_length/2],          # Right lower leg
            [0, 0, -lower_leg_length/2],          # Right foot
            [-hip_offset, 0, -torso_height/2],    # Left upper leg
            [0, 0, -upper_leg_length/2],          # Left lower leg
            [0, 0, -lower_leg_length/2]           # Left foot
        ]
        
        # Link orientations
        link_orientations = [
            [0, 0, 0, 1],  # Right upper leg
            [0, 0, 0, 1],  # Right lower leg
            [0, 0, 0, 1],  # Right foot
            [0, 0, 0, 1],  # Left upper leg
            [0, 0, 0, 1],  # Left lower leg
            [0, 0, 0, 1]   # Left foot
        ]
        
        # Parent indices (-1 for base, then link indices)
        link_parent_indices = [0, 0, 1, 0, 3, 4]  # 0 is base (torso)
        
        # Joint types
        joint_types = [
            p.JOINT_REVOLUTE,  # Right hip
            p.JOINT_REVOLUTE,  # Right knee
            p.JOINT_FIXED,     # Right foot (fixed to lower leg)
            p.JOINT_REVOLUTE,  # Left hip
            p.JOINT_REVOLUTE,  # Left knee
            p.JOINT_FIXED      # Left foot (fixed to lower leg)
        ]
        
        # Joint axes (rotation axes)
        joint_axes = [
            [1, 0, 0],  # Right hip (pitch)
            [1, 0, 0],  # Right knee (pitch)
            [0, 0, 0],  # Right foot (no rotation)
            [1, 0, 0],  # Left hip (pitch)
            [1, 0, 0],  # Left knee (pitch)
            [0, 0, 0]   # Left foot (no rotation)
        ]
        
        # Create the robot
        self.robot_id = p.createMultiBody(
            baseMass=torso_mass,
            baseCollisionShapeIndex=torso_collision,
            baseVisualShapeIndex=torso_visual,
            basePosition=self.initial_position,
            baseOrientation=[0, 0, 0, 1],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0, 0, 0]] * 6,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 6,
            linkParentIndices=link_parent_indices,
            linkJointTypes=joint_types,
            linkJointAxis=joint_axes
        )
        
        # Store joint indices for the controllable joints (exclude fixed joints)
        self.joint_indices = [0, 1, 3, 4]  # Right hip, right knee, left hip, left knee
        
        # Set joint limits and damping
        for joint_index in self.joint_indices:
            p.changeDynamics(
                self.robot_id,
                joint_index,
                linearDamping=0.1,
                angularDamping=0.1,
                jointDamping=0.1
            )
            # Set joint limits (approximately human-like)
            if joint_index in [0, 3]:  # Hip joints
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=0,
                    force=0
                )
            else:  # Knee joints
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=0,
                    force=0
                )
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            observation (np.array): Initial observation
            info (dict): Additional information
        """
        super().reset(seed=seed)
        
        self.episode_steps = 0
        
        # Reset robot position and orientation
        p.resetBasePositionAndOrientation(
            self.robot_id,
            self.initial_position,
            [0, 0, 0, 1]
        )
        
        # Reset joint positions to neutral stance
        for joint_index in self.joint_indices:
            p.resetJointState(self.robot_id, joint_index, 0, 0)
        
        # Let the robot settle
        for _ in range(10):
            p.stepSimulation()
        
        # Get initial observation
        observation = self._get_observation()
        self.prev_position = p.getBasePositionAndOrientation(self.robot_id)[0]
        
        info = {}
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (np.array): Actions to take (joint torques)
            
        Returns:
            observation (np.array): New observation
            reward (float): Reward for this step
            terminated (bool): Whether episode has ended
            truncated (bool): Whether episode was truncated
            info (dict): Additional information
        """
        self.episode_steps += 1
        
        # Apply actions (torque control)
        for i, joint_index in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.TORQUE_CONTROL,
                force=action[i]
            )
        
        # Step the simulation
        p.stepSimulation()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode should end
        terminated = self._is_terminated()
        truncated = self.episode_steps >= self.max_episode_steps
        
        info = {
            'episode_steps': self.episode_steps,
            'robot_height': observation[14],  # Z position of torso
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Get the current observation of the robot state.
        
        Returns:
            observation (np.array): Current state observation
        """
        # Get joint states
        joint_states = []
        for joint_index in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_index)
            joint_states.extend([joint_state[0], joint_state[1]])  # position, velocity
        
        # Get base position and orientation
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_orn_euler = p.getEulerFromQuaternion(base_orn)
        
        # Get base velocity
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Combine all observations
        observation = np.array(
            joint_states +                    # 8 values: 4 joint positions + 4 joint velocities
            list(base_orn_euler) +           # 3 values: roll, pitch, yaw
            list(base_pos) +                 # 3 values: x, y, z position
            list(base_vel),                  # 3 values: linear velocity
            dtype=np.float32
        )
        
        return observation
    
    def _calculate_reward(self):
        """
        Calculate the reward for the current state.
        
        Returns:
            reward (float): Calculated reward value
        """
        # Get current position
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # Reward for forward movement (positive x direction)
        if self.prev_position is not None:
            forward_progress = current_pos[0] - self.prev_position[0]
            forward_reward = forward_progress * FORWARD_REWARD_SCALE
        else:
            forward_reward = 0
        
        # Update previous position
        self.prev_position = current_pos
        
        # Penalty for falling (torso too low)
        height_penalty = 0
        if current_pos[2] < FALL_THRESHOLD:
            height_penalty = FALL_PENALTY
        
        # Penalty for excessive tilting
        _, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        stability_penalty = -abs(roll) * 5 - abs(pitch) * 5
        
        # Small reward for staying upright
        upright_reward = 1.0 if current_pos[2] > FALL_THRESHOLD else 0
        
        # Energy penalty (encourage efficient movement)
        # This would require storing previous actions, simplified for now
        energy_penalty = 0
        
        total_reward = forward_reward + height_penalty + stability_penalty + upright_reward + energy_penalty
        
        return total_reward
    
    def _is_terminated(self):
        """
        Check if the episode should be terminated.
        
        Returns:
            terminated (bool): Whether the episode should end
        """
        # Get current position
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # Episode ends if robot falls
        if current_pos[2] < FALL_THRESHOLD:
            return True
        
        return False
    
    def render(self, mode='human'):
        """
        Render the environment (handled automatically by PyBullet if GUI mode).
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            # Rendering is handled automatically by PyBullet GUI
            time.sleep(SIMULATION_TIMESTEP)
        
    def close(self):
        """
        Clean up the environment.
        """
        p.disconnect(self.physics_client)


# Test the environment
if __name__ == "__main__":
    env = RobotEnv(render_mode=True)
    
    print("Testing environment...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            obs, info = env.reset()
    
    env.close()
    print("Environment test completed!")