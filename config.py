"""
config.py

Author: Usman Siddiqi

Every tunable number of the project lives here: simulation, robot, terrain,
commands, rewards and PPO hyperparameters. The defaults are tuned for a
Ryzen 7 8845HS (8 cores / 16 threads) + RTX 4050 laptop:

  * physics runs on the CPU, batched over all hardware threads (MuJoCo rollout)
  * the neural networks (policy inference + PPO updates) run on the GPU

Change values here rather than in the code. Command line flags in main.py
override the most common ones (number of envs, iterations, device, ...).
"""

import os


class SimCfg:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot", "biped.xml")
    timestep = 0.005            # physics step [s] (200 Hz)
    decimation = 4              # physics steps per policy step -> 50 Hz control
    episode_length_s = 20.0     # episode timeout [s]

    @classmethod
    def control_dt(cls):
        return cls.timestep * cls.decimation


class RobotCfg:
    # Order must match the actuator order in robot/biped.xml
    joint_names = [
        "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle_pitch", "l_ankle_roll",
        "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle_pitch", "r_ankle_roll",
    ]
    # Nominal standing pose (slightly crouched) [rad]. Actions are offsets from it.
    default_joint_pos = [
        0.0, 0.0, -0.3, 0.6, -0.3, 0.0,
        0.0, 0.0, -0.3, 0.6, -0.3, 0.0,
    ]
    init_height = 0.79          # torso height above the ground at reset [m]
    nominal_height = 0.76       # torso height when standing in the default pose [m]
    action_scale = 0.3          # joint target = default + action_scale * action [rad]

    # Episode ends (robot "fell") when:
    min_height = 0.40           # torso lower than this above the terrain [m]
    max_tilt_gravity_z = -0.5   # projected gravity z above this (torso tilted > 60 deg)


class TerrainCfg:
    """
    The world is a grid of square tiles: one row per terrain type, one column
    per difficulty level. Every simulated robot lives in its own copy of the
    world, so robots never collide with each other and tiles can be shared.

      flat         infinite flat ground
      stairs_up    robot spawns at the bottom of a pit and must climb out
      stairs_down  robot spawns on top of a pyramid and must walk down
    """
    types = ["flat", "stairs_up", "stairs_down"]
    proportions = [0.2, 0.5, 0.3]   # fraction of robots assigned to each type
    tile_size = 10.0                # [m]
    platform_half = 1.0             # half size of the flat spawn platform [m]
    num_steps = 8                   # steps per staircase
    step_run = 0.30                 # step depth [m]
    num_levels = 10                 # curriculum levels
    step_height_min = 0.03          # step height at level 0 [m]
    step_height_max = 0.16          # step height at the last level [m]
    success_distance = 4.0          # distance from the tile centre that counts as "made it" [m]
    max_init_level = 1              # robots start on levels 0..max_init_level
    curriculum = True


class CommandCfg:
    lin_vel_x = [0.3, 0.9]          # commanded forward speed range [m/s]
    max_yaw_rate = 1.0              # [rad/s]
    heading_gain = 1.0              # yaw rate command = gain * heading error
    resample_time_s = 6.0           # speed (and heading on flat ground) is resampled this often
    gait_period = 0.7               # full left+right step cycle [s]
    stance_fraction = 0.55          # fraction of the cycle each foot should be on the ground


class DomainRandCfg:
    push_robots = True
    push_interval_s = 6.0
    max_push_vel = 0.4              # random horizontal velocity kick [m/s]
    init_joint_noise = 0.1          # [rad]
    init_vel_noise = 0.2            # [m/s], [rad/s]


class ObsCfg:
    # Height scan: terrain height sampled on a grid around the robot (in its heading frame)
    scan_x = [-0.36, -0.24, -0.12, 0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84]
    scan_y = [-0.24, -0.12, 0.0, 0.12, 0.24]
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    joint_vel_scale = 0.05
    height_scale = 5.0
    clip_actions = 10.0


class RewardCfg:
    """Reward = sum(weight * term) * control_dt. See BipedEnv._compute_reward."""
    tracking_sigma = 0.25
    swing_height_target = 0.09      # desired foot clearance above the terrain during swing [m]
    min_feet_distance = 0.14        # feet closer than this (sideways) are penalised [m]
    soft_joint_limit = 0.9          # fraction of the joint range that is penalty free

    weights = {
        # task
        "tracking_lin_vel": 1.5,    # follow the commanded forward speed
        "tracking_ang_vel": 0.75,   # follow the commanded yaw rate (heading)
        "alive": 0.15,              # not having fallen
        "gait_contact": 0.2,        # feet touch down / lift off in the commanded rhythm
        # style / regularisation
        "swing_height": -20.0,      # lift the swing foot to the target clearance
        "lin_vel_z": -0.5,          # bouncing
        "ang_vel_xy": -0.05,        # wobbling
        "orientation": -2.0,        # torso not upright
        "base_height": -20.0,       # crouching too low
        "hip_pos": -0.5,            # hip yaw / roll away from neutral
        "feet_distance": -5.0,      # feet too close together
        "foot_flat": -0.5,          # foot not parallel to the ground
        "feet_slip": -0.1,          # foot sliding while in contact
        "torques": -1.0e-5,
        "joint_vel": -5.0e-4,
        "joint_acc": -1.0e-7,
        "action_rate": -0.01,
        "joint_limits": -5.0,
        "termination": -100.0,      # one-off penalty for falling (scaled by dt like the rest)
    }


class PPOCfg:
    # rollout
    num_envs = 2048                 # parallel robots
    num_steps = 24                  # policy steps per robot per iteration
    max_iterations = 1500           # stairs are solved after ~800; the rest polishes the gait

    # networks
    actor_hidden = [512, 256, 128]
    critic_hidden = [512, 256, 128]
    init_noise_std = 1.0

    # PPO
    gamma = 0.99
    lam = 0.95
    clip_param = 0.2
    entropy_coef = 0.005
    value_loss_coef = 1.0
    num_epochs = 5
    num_minibatches = 4
    learning_rate = 1.0e-3
    desired_kl = 0.01               # learning rate is adapted to keep the policy change near this
    max_grad_norm = 1.0

    # bookkeeping
    save_interval = 100             # iterations between checkpoints
    log_interval = 10               # iterations between console prints


MODEL_SAVE_PATH = "models"
LOG_PATH = "logs"
