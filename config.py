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

    `rows` lists the terrains the policy TRAINS on. Parameters given as a
    (first level, last level) pair grow with the curriculum level; "run" is a
    range that is sampled randomly per tile. See terrain.py for the generators.
    The held-out test terrains live in EvalCfg below.
    """
    tile_size = 10.0                # [m]
    platform_half = 1.0             # half size of the flat spawn platform [m]
    obstacle_extent = 2.6           # stairs / slopes reach this far beyond the platform [m]
    num_levels = 10                 # curriculum levels
    success_distance = 4.0          # distance from the tile centre that counts as "made it" [m]
    max_init_level = 1              # robots start on levels 0..max_init_level
    curriculum = True

    rows = [
        dict(name="flat", kind="flat", weight=0.10),
        dict(name="rough", kind="rough", weight=0.10, height=(0.02, 0.08)),
        # two rows each, so every step height is seen with different tread depths
        dict(name="stairs_up", kind="stairs", weight=0.15, direction="up", rise=(0.03, 0.16), run=(0.26, 0.40)),
        dict(name="stairs_up", kind="stairs", weight=0.15, direction="up", rise=(0.03, 0.16), run=(0.26, 0.40)),
        dict(name="stairs_down", kind="stairs", weight=0.10, direction="down", rise=(0.03, 0.16), run=(0.26, 0.40)),
        dict(name="stairs_down", kind="stairs", weight=0.10, direction="down", rise=(0.03, 0.16), run=(0.26, 0.40)),
        dict(name="slope_up", kind="slope", weight=0.15, direction="up", slope=(0.05, 0.40), roughness=0.01),
        dict(name="slope_down", kind="slope", weight=0.15, direction="down", slope=(0.05, 0.40), roughness=0.01),
    ]
    flat_rows = [dict(name="flat", kind="flat", weight=1.0)]


class CommandCfg:
    lin_vel_x = [0.3, 0.9]          # commanded forward speed range [m/s]
    max_yaw_rate = 1.0              # [rad/s]
    heading_gain = 1.0              # yaw rate command = gain * heading error
    heading_noise = 0.5             # obstacles are approached up to this far from square-on [rad]
    resample_time_s = 6.0           # speed (and heading on flat ground) is resampled this often
    gait_period = 0.7               # full left+right step cycle [s]
    stance_fraction = 0.55          # fraction of the cycle each foot should be on the ground


class DomainRandCfg:
    """
    Randomisation during training, so the policy cannot rely on one exact
    simulator. `num_variants` copies of the model are built with different
    physical parameters and the robots are spread over them.
    """
    num_variants = 32
    friction = [0.5, 1.25]          # ground / foot friction coefficient
    payload = [-1.0, 3.0]           # extra torso mass [kg]
    com_offset = 0.03               # torso centre of mass shift, +- [m]
    motor_strength = [0.9, 1.1]     # scale on PD gains and torque limits

    push_robots = True
    push_interval_s = 5.0
    max_push_vel = 0.5              # random horizontal velocity kick [m/s]
    init_joint_noise = 0.1          # [rad]
    init_vel_noise = 0.2            # [m/s], [rad/s]

    # sensor noise (uniform, +-) added to the observation during training
    obs_noise = True
    noise_lin_vel = 0.1             # [m/s]
    noise_ang_vel = 0.2             # [rad/s]
    noise_gravity = 0.05
    noise_joint_pos = 0.01          # [rad]
    noise_joint_vel = 1.0           # [rad/s]
    noise_height_scan = 0.03        # [m]


class EvalCfg:
    """
    Held-out test conditions for `python main.py eval`. NOTHING in here is used
    for training: the terrains are different generators or parameter ranges,
    and the physics values lie outside the DomainRandCfg ranges.
    """
    unseen_terrains = [
        dict(name="flat", kind="flat", weight=0.0),
        # every step with its own random height and depth
        dict(name="irregular_stairs_up", kind="irregular_stairs", direction="up",
             rise=(0.04, 0.15), rise_jitter=(0.6, 1.25), run=(0.24, 0.45)),
        dict(name="irregular_stairs_down", kind="irregular_stairs", direction="down",
             rise=(0.04, 0.15), rise_jitter=(0.6, 1.25), run=(0.24, 0.45)),
        # steeper than anything in training: shorter treads (25 cm) and taller steps (up to 18 cm)
        dict(name="steep_stairs_up", kind="stairs", direction="up", rise=(0.04, 0.18), run=(0.25, 0.25)),
        dict(name="steep_stairs_down", kind="stairs", direction="down", rise=(0.04, 0.18), run=(0.25, 0.25)),
        # flights of three steps with landings in between
        dict(name="landing_stairs_up", kind="landing_stairs", direction="up", rise=(0.04, 0.16),
             run=0.275, landing=0.9, flights=2, steps_per_flight=3),
        # a ramp that turns into stairs
        dict(name="ramp_then_stairs", kind="ramp_stairs", slope=(0.05, 0.35), rise=(0.04, 0.15),
             run=0.30, ramp_length=1.2),
        # randomly scattered blocks to step on and off
        dict(name="scattered_blocks", kind="blocks", height=(0.03, 0.13), count=40),
        # rolling hills
        dict(name="waves", kind="waves", height=(0.05, 0.35), wavelength=2.5),
        # slopes and bumps beyond the training range
        dict(name="steep_slope_up", kind="slope", direction="up", slope=(0.30, 0.50)),
        dict(name="steep_slope_down", kind="slope", direction="down", slope=(0.30, 0.50)),
        dict(name="very_rough", kind="rough", height=(0.06, 0.12)),
    ]
    # run on the training terrains, with physics / commands the policy never experienced
    unseen_conditions = [
        # friction 0.4 is the physical limit for the steepest training slope (0.40), so do not go lower
        dict(name="slippery (friction 0.4)", physics=dict(friction=0.4)),
        dict(name="heavy payload (+6 kg)", physics=dict(payload=6.0)),
        dict(name="weak motors (80 %)", physics=dict(motor_strength=0.8)),
        dict(name="hard shoves (0.8 m/s every 3 s)", push=dict(max_push_vel=0.8, push_interval_s=3.0)),
        dict(name="diagonal approach (45 deg)", heading_offset=0.785),
        dict(name="all at once (0.45, +5 kg, 85 %, shoves)",
             physics=dict(friction=0.45, payload=5.0, motor_strength=0.85),
             push=dict(max_push_vel=0.6, push_interval_s=4.0)),
    ]


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
