"""
config.py

Author: Usman Siddiqi

Every tunable number of the project lives here: simulation, robot, the two
tasks (hike and sort), domain randomisation, rewards and PPO hyperparameters.
The defaults are tuned for a Ryzen 7 8845HS (8 cores / 16 threads) + RTX 4050
laptop:

  * physics runs on the CPU, batched over all hardware threads (MuJoCo rollout)
  * the neural networks (policy inference + PPO updates) run on the GPU

Change values here rather than in the code. Command line flags in main.py
override the most common ones (number of envs, iterations, device, ...).
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


class SimCfg:
    model_path = os.path.join(_HERE, "robot", "humanoid.xml")
    timestep = 0.005            # physics step [s] (200 Hz)
    decimation = 4              # physics steps per policy step -> 50 Hz control

    @classmethod
    def control_dt(cls):
        return cls.timestep * cls.decimation


class RobotCfg:
    # Order must match the actuator order in robot/humanoid.xml
    joint_names = [
        "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle_pitch", "l_ankle_roll",
        "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle_pitch", "r_ankle_roll",
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
    ]
    num_leg_joints = 12
    # Nominal standing pose (slightly crouched) [rad]. Actions are offsets from it.
    default_leg_pos = [
        0.0, 0.0, -0.3, 0.6, -0.3, 0.0,
        0.0, 0.0, -0.3, 0.6, -0.3, 0.0,
    ]
    # arms hanging next to the body, elbows slightly bent (hiking)
    default_arm_pos = [0.15, 0.12, 0.0, -0.5, 0.15, -0.12, 0.0, -0.5]
    # forearms pointing forward, palms facing each other (box handling)
    carry_arm_pos = [-0.25, 0.10, 0.0, -1.35, -0.25, -0.10, 0.0, -1.35]
    init_height = 0.79          # torso height above the ground at reset [m]
    nominal_height = 0.76       # torso height when standing in the default pose [m]
    leg_action_scale = 0.3      # joint target = default + action_scale * action [rad]
    arm_action_scale = 0.4

    # Episode ends (robot "fell") when:
    min_height = 0.40           # torso lower than this above the ground under it / its feet [m]
    max_tilt_gravity_z = -0.5   # projected gravity z above this (torso tilted > 60 deg)


class GaitCfg:
    period = 0.7                # full left+right step cycle [s]
    stance_fraction = 0.55      # fraction of the cycle each foot should be on the ground
    stand_speed = 0.1           # below this commanded speed the robot should stand with both feet down


# ======================================================================== HIKE
class HikeCfg:
    """
    Task 1: hiking. The robot follows a winding mountain trail from one end of a
    tile to the other. The trail is a raised path made of rock columns: fall off
    the side (or into a gap) and the episode is over.

    The world is a set of tiles: one row per kind of trail ("biome"), one column
    per difficulty level. Every simulated robot lives in its own copy of one
    tile, so robots never collide with each other.

    `rows` lists what the policy TRAINS on. `sections` are the building blocks a
    trail of that row is assembled from (see terrain.py). The held-out test
    trails live in EvalCfg below.
    """
    episode_length_s = 25.0
    tile_length = 12.0              # along the trail (x) [m]
    tile_width = 8.0                # across (y) [m]
    cell = 0.125                    # trails are built from square rock columns of this size [m]
    start_x = -5.4                  # robots start here (tile coordinates, the tile centre is 0) [m]
    goal_x = 5.25                   # ... and have made it when they get here
    base_elevation = 1.0            # height of the trail above the surrounding ground at the start [m]
    elevation_range = (0.6, 2.6)    # the trail climbs / descends within these heights [m]
    num_levels = 10                 # curriculum levels
    max_init_level = 1              # robots start on levels 0..max_init_level
    curriculum = True
    lookahead = 0.75                # the robot steers towards the trail centre this far ahead [m]
    off_trail_margin = 0.6          # further than this beyond the trail edge = left the trail [m]

    # Difficulty: every pair is (level 0, last level)
    winding = (0.2, 1.6)            # how far the trail swings left / right [m]
    half_width = (1.5, 0.6)         # half the width of the trail [m]
    ridge_half_width = (0.7, 0.25)  # ... on ridge sections
    step_rise = (0.04, 0.20)        # rock steps [m]
    rough_height = (0.02, 0.10)     # bumps on rough sections, peak to peak [m]
    stone_gap = (0.0, 0.25)         # gap between stepping stones [m]
    stone_half_width = (0.6, 0.25)
    stone_zigzag = (0.0, 0.3)       # stepping stones alternate this far left / right of the trail [m]
    boulders = (1, 5)               # boulders per boulder section
    boulder_clearance = (0.7, 0.42) # free space between the trail centre and a boulder [m]

    rows = [
        dict(name="flat", kind="flat", weight=0.20),
        dict(name="rocky", kind="trail", weight=0.125, sections=["rough", "steps", "flat"]),
        dict(name="climb", kind="trail", weight=0.125, sections=["steps_up", "steps_up", "steps_down", "steps_down", "flat"]),
        dict(name="stones", kind="trail", weight=0.125, sections=["stones", "stones", "flat", "rough"]),
        dict(name="ridge", kind="trail", weight=0.125, sections=["ridge", "boulders", "ridge", "rough"]),
        # the real thing: everything mixed (two rows, so there are twice as many different trails)
        dict(name="trail", kind="trail", weight=0.15,
             sections=["rough", "steps", "steps_up", "steps_down", "stones", "ridge", "boulders"]),
        dict(name="trail", kind="trail", weight=0.15,
             sections=["rough", "steps", "steps_up", "steps_down", "stones", "ridge", "boulders"]),
    ]
    flat_rows = [dict(name="flat", kind="flat", weight=1.0)]

    # Commands. On a trail the robot is steered towards the trail ahead: it keeps walking and turns
    # in an arc. On flat ground half of the robots practise exactly that with random directions, the
    # other half gets random velocity commands (forwards / backwards, sideways, turning on the spot),
    # so the policy can move in any direction - the sorting task builds on that.
    speed = [0.4, 0.9]              # commanded walking speed [m/s]
    max_side_speed = 0.3            # [m/s]
    max_back_speed = 0.3            # [m/s]
    max_yaw_rate = 1.0              # [rad/s]
    heading_gain = 1.5              # yaw rate command = gain * heading error
    min_turn_speed = 0.6            # share of the speed that is kept while facing the wrong way
                                    # (much less and standing still is the lazy answer to a big turn)
    resample_time_s = 5.0           # on flat ground: new command this often
    stand_probability = 0.1         # on flat ground: share of commands that are "stand still"
    velocity_mode = 0.5             # on flat ground: share of random velocity commands

    # Height scan: terrain height sampled on a grid around the robot (in its heading frame)
    scan_x = [-0.30 + 0.09 * i for i in range(16)]      # -0.30 .. 1.05 m
    scan_y = [-0.40 + 0.10 * i for i in range(9)]       # -0.40 .. 0.40 m


class DomainRandCfg:
    """
    Randomisation during training, so the policy cannot rely on one exact
    simulator. Several copies of every model are built with different physical
    parameters and the robots are spread over them.
    """
    variants_per_tile = 6
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
    noise_object_pos = 0.01         # [m] (sorting: box and table positions)


class EvalCfg:
    """
    Held-out test conditions for `python main.py eval`. NOTHING in here is used
    for training: the trails are built from sections the policy never saw (or
    lie outside the training ranges), and the physics values lie outside the
    DomainRandCfg ranges.
    """
    unseen_rows = [
        dict(name="flat", kind="flat", weight=0.0),
        # logs across the trail to step over
        dict(name="logs", kind="trail", sections=["logs", "logs", "rough"]),
        # the left and right half of the trail at different heights
        dict(name="terraces", kind="trail", sections=["terraces", "terraces", "flat"]),
        # loose rubble: smaller and taller bumps than the training "rough"
        dict(name="rubble", kind="trail", sections=["rubble"]),
        # regular staircases, steeper than the irregular rock steps of training
        dict(name="staircase", kind="trail", sections=["stairs_up", "stairs_down"], step_rise=(0.05, 0.22)),
        # switchbacks: swings left / right 60 % further than any training trail
        dict(name="switchbacks", kind="trail", sections=["rough", "ridge", "steps"], winding=(0.6, 2.5)),
        # all the training sections plus the new ones, on trails it has never walked
        dict(name="new_trail", kind="trail",
             sections=["rough", "steps", "stones", "ridge", "boulders", "logs", "terraces", "rubble", "stairs_up"]),
    ]
    unseen_seed = 1234
    # the training trails, with physics / disturbances the policy never experienced
    unseen_conditions = [
        dict(name="slippery (friction 0.4)", physics=dict(friction=0.4)),
        dict(name="heavy backpack (+6 kg)", physics=dict(payload=6.0)),
        dict(name="weak motors (80 %)", physics=dict(motor_strength=0.8)),
        dict(name="hard shoves (0.8 m/s every 3 s)", push=dict(max_push_vel=0.8, push_interval_s=3.0)),
        dict(name="all at once (0.45, +5 kg, 85 %, shoves)",
             physics=dict(friction=0.45, payload=5.0, motor_strength=0.85),
             push=dict(max_push_vel=0.6, push_interval_s=4.0)),
    ]
    # the sorting room, with boxes / physics / disturbances the policy never experienced
    sort_unseen_conditions = [
        dict(name="heavier boxes (6.5 - 8 kg)", box_mass=(6.5, 8.0)),
        dict(name="feather-light boxes (0.3 - 1 kg)", box_mass=(0.3, 1.0)),
        dict(name="slippery floor (friction 0.4)", physics=dict(friction=0.4)),
        dict(name="weak motors (80 %)", physics=dict(motor_strength=0.8)),
        dict(name="heavy backpack (+6 kg)", physics=dict(payload=6.0)),
        dict(name="shoves (0.6 m/s every 4 s)", push=dict(max_push_vel=0.6, push_interval_s=4.0)),
    ]


class ObsCfg:
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    joint_vel_scale = 0.05
    height_scale = 5.0
    clip_actions = 10.0


class RewardCfg:
    """Reward = sum(weight * term) * control_dt. See RobotEnv._common_terms and the task classes."""
    tracking_sigma = 0.25
    tracking_sigma_yaw = 0.5        # wider: a robot that cannot turn yet still feels which way is better
    swing_height_target = 0.09      # desired foot clearance above the terrain during swing [m]
    min_feet_distance = 0.14        # feet closer than this (sideways) are penalised [m]
    soft_joint_limit = 0.9          # fraction of the joint range that is penalty free

    # shared by both tasks: posture, feet, effort
    common = {
        "alive": 0.15,              # not having fallen
        "gait_contact": 0.2,        # feet touch down / lift off in the commanded rhythm
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
    hike = {
        "tracking_lin_vel": 1.5,    # follow the commanded velocity
        "tracking_ang_vel": 0.75,   # follow the commanded yaw rate
        "heading": 0.5,             # face along the trail (or the commanded direction)
        "arm_pos": -0.05,           # arms far from the relaxed pose (they are free to swing for balance)
    }


# ======================================================================== SORT
class SortCfg:
    """
    Task 2: sorting. A box appears on the pickup table. The robot walks up to it,
    squeezes it between its palms, lifts it, carries it to the table whose colour
    matches the box, puts it down there and lets go. Then the next box appears.
    Boxes differ in weight (and size) and the robot is not told the weight: it
    has to squeeze harder and lean back further for the heavy ones.
    """
    episode_length_s = 30.0
    colors = [("red", (0.86, 0.22, 0.20)), ("green", (0.22, 0.66, 0.30)), ("blue", (0.20, 0.42, 0.90))]
    shuffle_colors = True           # which table has which colour changes every episode
    table_height = 0.70             # [m]
    table_half = 0.25               # half size of the square table tops [m]
    table_distance = 1.25           # tables stand this far from the middle of the room [m]
    # directions of the pickup table and the drop-off tables, seen from the middle [deg] (multiples of 90)
    pickup_angle = 0.0
    drop_angles = [90.0, 180.0, 270.0]
    box_inset = 0.13                # boxes are picked up / put down this far from the table edge [m]

    # boxes: depth, width (between the palms), height [m]; the robot weighs 28.5 kg
    box_sizes = [[0.20, 0.26, 0.20], [0.18, 0.24, 0.18], [0.22, 0.28, 0.22], [0.20, 0.24, 0.16], [0.18, 0.28, 0.22]]
    box_mass = [1.5, 6.0]           # [kg]
    num_box_masses = 6              # different masses within that range
    num_robot_variants = 8          # robot / floor physics variants (domain randomisation)
    box_friction = 1.0
    # Contact model. With MuJoCo's default (pyramidal, soft) friction a squeezed box creeps out of the
    # hands within seconds. So: the exact elliptic friction cone everywhere, and a stiff, grippy
    # contact between the palms and the box. (A high global impratio would also stop the creep,
    # but it changes how the feet grip the floor and the hiking policy starts to stumble.)
    elliptic_cone = True
    impratio = 2.0
    palm_friction = 1.6
    palm_solreffriction = [0.01, 1.0]
    palm_solimp = [0.98, 0.99, 0.001, 0.5, 2.0]

    stand_off = 0.34                # torso <-> box distance for picking up / putting down [m]
    lift_height = 0.08              # a box counts as picked up when it is this far above the table [m]
    place_margin = 0.07             # a placed box must be at least this far inside the table edge [m]
    place_hold_s = 0.3              # ... lie still for this long
    release_distance = 0.05         # ... with both palms at least this far from it [m]
    start_near = 0.5                # share of episodes that start in front of the pickup table (the rest: anywhere)

    # walking between the tables
    speed = 0.5                     # [m/s]
    approach_gain = 1.5             # slows down on arrival: speed = gain * distance [1/s]
    max_side_speed = 0.3
    max_back_speed = 0.2
    max_yaw_rate = 1.0
    heading_gain = 1.5

    # Guidance. The sorting policy starts as a copy of the hiking policy, which walks wherever a
    # velocity command tells it to. At first a "guide" (a few lines of code that know the rules)
    # gives it such commands: to the box, then to the table of the box's colour. As soon as the
    # robot sorts boxes that way, the guide is taken away from more and more robots, which then
    # only see the colour of the box and where the three coloured tables are - until no robot
    # is guided any more and the policy does the sorting by itself.
    guide_start = 1.0               # share of guided robots at the start of training
    guide_step = 0.05               # ... is reduced by this much
    guide_every = 10                # ... every so many iterations
    guide_threshold = 1.0           # ... while robots sort at least this many boxes per episode on average

    # The task reward is a progress score:  5 * boxes sorted + stage + progress within the stage (0..1)
    #   stage 0  fetch    walk to the box, palms to its two sides
    #   stage 1  lift     squeeze and lift it off the table
    #   stage 2  carry    carry it to the table of its colour
    #   stage 3  lower    lower it onto that table
    #   stage 4  release  let go
    # Finishing a stage is always worth more than lingering in it.
    rewards = {
        "stage": 1.0,
        "tracking_lin_vel": 1.0,    # walking where it should (see "Guidance": the robot may not be told)
        "tracking_ang_vel": 0.5,
        "heading": 0.3,             # facing where it is going / the table it is at
        "box_vel": -0.5,            # held box moving relative to the torso
        "arm_pos": 0.0,
        "sorted": 250.0,            # one-off bonus per box sorted (scaled by dt like the rest)
        "failed": -150.0,           # one-off: box dropped on the floor or put on the wrong table
    }


class PPOCfg:
    # rollout
    num_envs = 2048                 # parallel robots
    num_steps = 24                  # policy steps per robot per iteration
    max_iterations = 3000

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

    # starting from another policy (sorting starts from the hiking policy)
    critic_warmup = 20              # iterations in which only the value function learns
    transfer_arm_std = 0.6          # exploration noise of the arm joints after the transfer

    # bookkeeping
    save_interval = 50              # iterations between checkpoints
    log_interval = 10               # iterations between console prints


MODEL_SAVE_PATH = "models"
LOG_PATH = "logs"
