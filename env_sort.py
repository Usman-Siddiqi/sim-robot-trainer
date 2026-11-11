"""
env_sort.py

Author: Usman Siddiqi

Task 2 - sorting boxes by colour. The room has a pickup table and three
drop-off tables (red, green, blue - which is which changes every episode). A
box of one of the three colours lies on the pickup table. The robot has to

    walk to it -> palms against its sides, under the rim -> lift it -> carry it
    to the table of the same colour -> put it down -> let go

and then the next box appears. The boxes are crates with a rim around the top
that rests on the palms. They weigh 0.5 - 6 kg (the robot: 28.5 kg) and come in
several sizes. The robot is not told the weight; it feels it through its joints
and has to hold on and lean back further for the heavy ones.

The finished policy is told nothing about where to go: it sees the colour of
the box and where the red, the green and the blue table are, and works out the
rest. A box that is dropped or put on the wrong table costs a penalty and is
replaced by a new one; falling over ends the episode.

Training starts from the hiking policy, so the robot can already walk wherever
a velocity command sends it. At first a guide supplies such commands (to the
box, then to the right table); it is withdrawn step by step as the robot gets
better - see "Guidance" in SortCfg.

Observation (253): the hiking observation (base velocity, angular velocity,
gravity direction, joint positions / velocities, previous action, gait clock,
velocity command - all zero for robots without a guide - and height scan), then,
relative to the torso in its heading frame: the box (position, orientation,
velocity, size, colour), the palms (position, contact force), the pickup table
and the three coloured tables, and how many boxes were already sorted.
"""

import mujoco
import numpy as np

from config import DomainRandCfg, GaitCfg, HikeCfg, ObsCfg, RewardCfg, RobotCfg, SimCfg, SortCfg
from env_base import RobotEnv, quat_rotate_inverse, wrap_to_pi

NUM_STAGES = 5
STAGE_NAMES = ["fetch", "lift", "carry", "lower", "release"]


def _quat_to_axes(q):
    """Columns of the rotation matrix of quaternions (N,4) -> x, y, z axes, each (N,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ax = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
    ay = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1)
    az = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return ax, ay, az


def build_scene(box_size):
    """Robot + floor + the four tables + one box of the given size -> MjModel."""
    cfg = SortCfg
    spec = mujoco.MjSpec.from_file(SimCfg.model_path)
    if cfg.elliptic_cone:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = cfg.impratio
    world = spec.worldbody
    world.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                   material="grid", contype=0, conaffinity=1)
    top, half = cfg.table_height, cfg.table_half
    for k, angle in enumerate([cfg.pickup_angle] + list(cfg.drop_angles)):
        cx, cy = cfg.table_distance * np.cos(np.radians(angle)), cfg.table_distance * np.sin(np.radians(angle))
        rgba = [0.55, 0.57, 0.60, 1]
        world.add_geom(name=f"table{k}_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, top - 0.02],
                       size=[half, half, 0.02], rgba=rgba, contype=0, conaffinity=1, friction=[1.0, 0.005, 0.0001])
        world.add_geom(name=f"table{k}_leg", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, (top - 0.04) / 2],
                       size=[0.05, 0.05, (top - 0.04) / 2], rgba=[0.3, 0.31, 0.33, 1], contype=0, conaffinity=1)
        # coloured mat on the floor, so the zones can be told apart at a glance (no collisions)
        world.add_geom(name=f"table{k}_mat", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, 0.001],
                       size=[half + 0.2, half + 0.2, 0.001], rgba=rgba, contype=0, conaffinity=0)
    box = world.add_body(name="box", pos=[0, 0, 3.0])
    box.add_freejoint()
    hx, hy, hz = [s / 2 for s in box_size]
    box.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], mass=3.0,
                 contype=1, conaffinity=1, condim=4, friction=[cfg.box_friction, 0.02, 0.0001], rgba=[0.8, 0.2, 0.2, 1])
    # the rim: a lid that sticks out on the left and right side; it rests on the palms when the box is
    # carried (no rim at the front and back, where it would bump into the robot's chest)
    rim, thick = cfg.rim_width, cfg.rim_thickness
    box.add_geom(name="box_rim", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, hz - thick / 2], size=[hx, hy + rim, thick / 2],
                 mass=0.1, contype=1, conaffinity=1, condim=4, friction=[cfg.box_friction, 0.02, 0.0001], rgba=[0.7, 0.18, 0.18, 1])
    # pressure sensor under the box: how much of its weight still rests on a table
    box.add_site(name="box_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -hz], size=[hx + 0.01, hy + 0.01, 0.015], rgba=[0, 0, 0, 0])
    spec.add_sensor(name="box_bottom", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="box_bottom")
    for palm in ("l_palm", "r_palm"):
        mu = cfg.palm_friction
        spec.add_pair(geomname1=palm, geomname2="box", condim=4, friction=[mu, mu, 0.02, 0.0001, 0.0001],
                      solreffriction=cfg.palm_solreffriction, solimp=cfg.palm_solimp)
    model = spec.compile()
    model.opt.timestep = SimCfg.timestep
    return model


class SortEnv(RobotEnv):
    task = "sort"
    episode_length_s = SortCfg.episode_length_s
    default_arm_pos = RobotCfg.carry_arm_pos
    progress_keys = (("sorted", "sorted"), ("stage", "stage"), ("drop", "dropped"), ("wrong", "wrong_table"),
                     ("guided", "guided"), ("kg", "max_box_mass"))

    def __init__(self, num_envs, num_threads=None, seed=0, randomize=True, physics=None, push=None,
                 box_mass=None, start_near=None, guide=0.0, mass_level=None):
        """
        Args (see RobotEnv for the rest):
            box_mass: (lowest, highest) box mass [kg] instead of SortCfg.box_mass
            start_near: share of episodes starting in front of the pickup table instead of SortCfg.start_near
            guide: share of robots that get velocity commands from the guide (0 = the policy is on its
                   own, which is how a finished policy is used; training starts at SortCfg.guide_start)
            mass_level: only the lightest `mass_level` box masses are used (default: all; training
                        starts at SortCfg.mass_start)
        """
        super().__init__(num_envs, num_threads, seed, randomize, physics, push)
        cfg = SortCfg
        n = self.num_envs
        self.start_near = cfg.start_near if start_near is None else start_near
        self.guide_probability = float(guide)

        # ---- scenes = box sizes; variants = robot physics x box mass
        scenes = [build_scene(size) for size in cfg.box_sizes]
        self.box_half_of_scene = np.array(cfg.box_sizes) / 2
        lo, hi = box_mass or cfg.box_mass
        self.box_masses = np.linspace(lo, hi, cfg.num_box_masses)
        self.mass_level = int(np.clip(mass_level or cfg.num_box_masses, 1, cfg.num_box_masses))
        if physics:
            robots = [dict(physics)]
        elif randomize:
            robots = [self._random_physics() for _ in range(cfg.num_robot_variants)]
        else:
            robots = [{}]
        self.num_robot_variants = len(robots)
        variants = [[self._make_variant(scene, box_mass=mass, **robot) for robot in robots for mass in self.box_masses]
                    for scene in scenes]
        self._init_sim(scenes, variants=variants)
        m = scenes[0]
        self._box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self._rim_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box_rim")
        self._table_geoms = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"table{k}_{part}") for part in ("top", "mat")]
                             for k in range(4)]
        nu = self.num_actions
        self._box_qpos = slice(1 + 7 + nu, 1 + 14 + nu)
        self._box_qvel = slice(1 + self.nq + 6 + nu, 1 + self.nq + 12 + nu)

        # ---- room layout: table 0 = pickup, 1..3 = drop-off
        angles = np.radians([cfg.pickup_angle] + list(cfg.drop_angles))
        self.table_dir = np.stack([np.cos(angles), np.sin(angles)], axis=1)      # from the middle outwards
        self.table_xy = cfg.table_distance * self.table_dir
        # where a box is picked up / should be put down on every table
        self.place_xy = self.table_xy - self.table_dir * (cfg.table_half - cfg.box_inset)
        self.colors = np.array([rgb for _, rgb in cfg.colors])

        # ---- task state
        self.robot_variant = self.rng.integers(0, self.num_robot_variants, n)
        self.box_mass_index = np.zeros(n, dtype=np.int64)
        self.box_half = np.zeros((n, 3))
        self.box_color = np.zeros(n, dtype=np.int64)
        self.table_color = np.tile(np.arange(3), (n, 1))        # colour of drop-off table 1..3
        self.target = np.ones(n, dtype=np.int64)                # table (1..3) the current box belongs on
        self.sorted_count = np.zeros(n, dtype=np.int64)
        self.stage = np.zeros(n, dtype=np.int64)
        self.stage_progress = np.zeros(n)
        self.max_stage = np.zeros(n, dtype=np.int64)
        self.placed_steps = np.zeros(n, dtype=np.int64)
        self.wrong_steps = np.zeros(n, dtype=np.int64)
        self.just_sorted = np.zeros(n, dtype=bool)
        self.drop_count = np.zeros(n)
        self.wrong_count = np.zeros(n)
        self.table_force = np.zeros(n)                          # force on the bottom of the box [N]
        self.commands = np.zeros((n, 3))                        # forward speed, sideways speed, yaw rate
        self.guided = np.zeros(n, dtype=bool)

        self.num_scan = len(HikeCfg.scan_x) * len(HikeCfg.scan_y)
        self.num_obs = 71 + 3 + self.num_scan + (3 + 6 + 3 + 3 + 3 + 6 + 2 + 2 + 6 + 1)
        noise = DomainRandCfg.noise_object_pos
        self._noise_scale = np.concatenate(self._proprioception_noise() + [
            np.zeros(3), np.full(self.num_scan, DomainRandCfg.noise_height_scan * ObsCfg.height_scale),
            np.full(3, noise), np.full(6, 0.02), np.full(3, 0.05), np.full(3, 0.005), np.zeros(3),
            np.full(6, noise), np.zeros(2), np.full(2, noise), np.full(6, noise), np.zeros(1),
        ]).astype(np.float32)
        weights = {**RewardCfg.common, **cfg.rewards}
        self._init_rewards(weights)

    # ---------------------------------------------------------------- models
    def _task_variant(self, m, box_mass=None):
        if box_mass is None:
            return
        box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        hx, hy, hz = m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")]
        m.body_mass[box] = box_mass
        m.body_inertia[box] = box_mass / 3.0 * np.array([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])

    def _new_box(self, ids):
        """Put a fresh box (random colour, weight and size) on the pickup table of the given robots."""
        cfg, rng, n = SortCfg, self.rng, ids.size
        self.env_scene[ids] = rng.integers(0, len(self.scene_models), n)
        self.box_mass_index[ids] = rng.integers(0, self.mass_level, n)
        self.env_variant[ids] = self.robot_variant[ids] * self.box_masses.size + self.box_mass_index[ids]
        self._assign_models(ids)
        self.box_half[ids] = self.box_half_of_scene[self.env_scene[ids]]
        self.box_color[ids] = rng.integers(0, 3, n)
        self.target[ids] = 1 + np.argmax(self.table_color[ids] == self.box_color[ids, None], axis=1)

        yaw = np.arctan2(self.table_dir[0, 1], self.table_dir[0, 0]) + rng.uniform(-0.15, 0.15, n)
        qpos = np.zeros((n, 7))
        qpos[:, 0:2] = self.place_xy[0] + rng.uniform(-0.02, 0.02, (n, 2))
        qpos[:, 2] = cfg.table_height + self.box_half[ids, 2] + 0.002
        qpos[:, 3], qpos[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
        self.state[ids, self._box_qpos] = qpos
        self.state[ids, self._box_qvel] = 0.0
        self.table_force[ids] = 9.81 * self.box_masses[self.box_mass_index[ids]]
        self.placed_steps[ids] = 0
        self.wrong_steps[ids] = 0

    def write_data(self, index, data):
        """Copy the state of robot `index` into an MjData and colour its box and tables (for rendering)."""
        model = self._models[index]
        model.geom_rgba[self._box_geom, :3] = self.colors[self.box_color[index]]
        model.geom_rgba[self._rim_geom, :3] = 0.8 * self.colors[self.box_color[index]]
        for k in range(1, 4):
            for g in self._table_geoms[k]:
                model.geom_rgba[g, :3] = self.colors[self.table_color[index, k - 1]]
        super().write_data(index, data)

    # ---------------------------------------------------------------- task
    def _reset_envs(self, ids):
        cfg, rng, n = SortCfg, self.rng, ids.size
        if cfg.shuffle_colors:
            self.table_color[ids] = np.argsort(rng.random((n, 3)), axis=1)
        self.sorted_count[ids] = 0
        self.drop_count[ids] = 0
        self.wrong_count[ids] = 0
        self.max_stage[ids] = 0
        self.guided[ids] = rng.random(n) < self.guide_probability
        self._new_box(ids)

        # the robot: in front of the pickup table, or anywhere in the middle of the room
        near = rng.random(n) < self.start_near
        out = self.table_dir[0]
        facing = np.arctan2(out[1], out[0])
        stand = self.place_xy[0] - out * cfg.stand_off
        r, a = 0.6 * np.sqrt(rng.random(n)), rng.uniform(-np.pi, np.pi, n)
        x = np.where(near, stand[0] - out[0] * rng.uniform(0.0, 0.25, n) + rng.uniform(-0.08, 0.08, n), r * np.cos(a))
        y = np.where(near, stand[1] - out[1] * rng.uniform(0.0, 0.25, n) + rng.uniform(-0.08, 0.08, n), r * np.sin(a))
        yaw = np.where(near, facing + rng.uniform(-0.3, 0.3, n), rng.uniform(-np.pi, np.pi, n))
        self._robot_reset_state(ids, x, y, RobotCfg.init_height, yaw)

    def _read_task_sensors(self, sensors):
        self.table_force = sensors.mean(axis=1)[:, 0]

    def _update_task(self):
        """Work out, from the physical state alone, which stage every robot is in and how far along."""
        cfg = SortCfg
        n = self.num_envs
        ar = np.arange(n)
        self.box_pos = self.state[:, self._box_qpos][:, 0:3]
        self.box_quat = self.state[:, self._box_qpos][:, 3:7]
        self.box_vel = self.state[:, self._box_qvel][:, 0:3]
        bx, by, bz = _quat_to_axes(self.box_quat)
        self.box_axes = (bx, by)
        hx, hy, hz = self.box_half.T

        # where the palms belong: on the two faces of the box that point left / right as seen by the
        # robot, with their upper edge just under the rim
        left = np.stack([-np.sin(self.yaw), np.cos(self.yaw), np.zeros(n)], axis=1)
        reach = np.abs((left * bx).sum(1)) * hx + np.abs((left * by).sum(1)) * hy + np.abs((left * bz).sum(1)) * hz
        middle = self.box_pos + bz * (hz - cfg.rim_thickness - 0.055)[:, None]
        self.grip_dist = np.stack([
            np.linalg.norm(self.grip_pos[:, 0] - (middle + left * reach[:, None]), axis=1),
            np.linalg.norm(self.grip_pos[:, 1] - (middle - left * reach[:, None]), axis=1)], axis=1)
        gripped = (self.grip_dist < cfg.grip_distance).all(axis=1)

        # the box relative to the tables
        half_up = np.abs(bx[:, 2]) * hx + np.abs(by[:, 2]) * hy + np.abs(bz[:, 2]) * hz
        self.box_above = self.box_pos[:, 2] - half_up - cfg.table_height          # bottom of the box above the table tops
        offset = np.abs(self.box_pos[:, None, 0:2] - self.table_xy[None, :, :]).max(axis=2)     # (N, 4)
        over = offset <= cfg.table_half
        over_pickup = over[:, 0]
        over_target = offset[ar, self.target] <= cfg.table_half - cfg.place_margin
        resting = (self.box_above < 0.015) & (self.box_above > -0.03)
        self.on_target = over_target & resting
        over_wrong = over[:, 1:].any(axis=1) & ~over[ar, self.target]
        self.on_wrong = over_wrong & resting
        self.dropped = self.box_pos[:, 2] < cfg.table_height - 0.15

        # where the robot should stand: in front of the box (fetch / lift), else in front of the right table
        goal_dir = np.where((self.stage < 2)[:, None], self.table_dir[0], self.table_dir[self.target])
        to_place = self.place_xy[self.target] - self.box_pos[:, 0:2]
        place_dist = np.linalg.norm(to_place, axis=1)

        # ---- stage and progress within it
        fetch_stand = self.box_pos[:, 0:2] - self.table_dir[0] * cfg.stand_off
        fetch_dist = np.linalg.norm(fetch_stand - self.base_pos[:, 0:2], axis=1)
        stage = np.zeros(n, dtype=np.int64)
        progress = 0.3 * np.exp(-fetch_dist / 0.5) + 0.7 * np.exp(-self.grip_dist.sum(axis=1) / 0.2)
        lifting = gripped & over_pickup & (self.box_above < cfg.lift_height)
        stage[lifting] = 1
        # before the box leaves the table: the share of its weight the hands already carry
        weight = 9.81 * self.box_masses[self.box_mass_index]
        carried = np.clip(1.0 - self.table_force / weight, 0.0, 1.0)
        progress[lifting] = 0.5 * carried[lifting] + 0.5 * np.clip(self.box_above[lifting] / cfg.lift_height, 0.0, 1.0)
        carrying = gripped & ~lifting & ~over_target
        stage[carrying] = 2
        progress[carrying] = 1.0 - np.clip(place_dist[carrying] / 2.6, 0.0, 1.0)
        lowering = gripped & over_target & ~resting
        stage[lowering] = 3
        progress[lowering] = 1.0 - np.clip(self.box_above[lowering] / 0.12, 0.0, 1.0)
        stage[self.on_target] = 4
        progress[self.on_target] = np.clip((self.grip_dist[self.on_target].min(axis=1) - cfg.grip_distance) /
                                           (cfg.release_distance + 0.01), 0.0, 1.0)
        self.stage, self.stage_progress = stage, progress
        self.max_stage = np.maximum(self.max_stage, stage)

        # ---- the guide: velocity command towards where the robot should stand
        goal_dir = np.where((stage < 2)[:, None], self.table_dir[0], self.table_dir[self.target])
        stand = np.where((stage < 2)[:, None], fetch_stand, self.place_xy[self.target] - goal_dir * cfg.stand_off)
        to_stand = stand - self.base_pos[:, 0:2]
        dist = np.linalg.norm(to_stand, axis=1)
        speed = np.clip(cfg.approach_gain * dist, 0.0, cfg.speed)
        speed[dist < 0.03] = 0.0
        travel = np.arctan2(to_stand[:, 1], to_stand[:, 0])
        table = np.arctan2(goal_dir[:, 1], goal_dir[:, 0])
        err = wrap_to_pi(travel - self.yaw)
        ahead = np.cos(err)
        far = dist > 0.5
        # far away: walk there in an arc, like on a trail. Close: face the table and shuffle into place.
        self.commands[:, 0] = np.where(far, speed * np.maximum(ahead, HikeCfg.min_turn_speed),
                                       np.clip(speed * ahead, -cfg.max_back_speed, None))
        side = speed * np.sin(err) * np.where(far, np.maximum(ahead, 0.0), 1.0)
        self.commands[:, 1] = np.clip(side, -cfg.max_side_speed, cfg.max_side_speed)
        face_err = wrap_to_pi(np.where(far, travel, table) - self.yaw)
        self.commands[:, 2] = np.clip(cfg.heading_gain * face_err, -cfg.max_yaw_rate, cfg.max_yaw_rate)
        self.face_err = face_err
        self.commands[np.abs(self.commands) < 0.03] = 0.0
        self.standing = (np.abs(self.commands) < GaitCfg.stand_speed).all(axis=1)
        self.stand_dist = dist
        self.gripped = gripped
        # a box still lying on the pickup table should not be shoved around or tipped over
        self.box_shove = (stage == 0) * ((self.box_vel[:, 0:2] ** 2).sum(axis=1) + 4.0 * (1.0 - np.abs(bz[:, 2])))

    def _check_termination(self):
        cfg = SortCfg
        fell = (self.base_pos[:, 2] < RobotCfg.min_height) | (self.projected_gravity[:, 2] > RobotCfg.max_tilt_gravity_z)

        # a box counts as sorted when it lies still on the right table with both hands off it
        released = (self.grip_dist > cfg.release_distance).all(axis=1)
        still = np.linalg.norm(self.box_vel, axis=1) < 0.1
        self.placed_steps = np.where(self.on_target & released & still, self.placed_steps + 1, 0)
        self.just_sorted = self.placed_steps >= int(cfg.place_hold_s / self.dt)
        self.wrong_steps = np.where(self.on_wrong & ~self.gripped, self.wrong_steps + 1, 0)
        wrong = self.wrong_steps >= int(cfg.place_hold_s / self.dt)
        dropped = self.dropped & ~fell
        wrong &= ~dropped & ~fell
        self.failed = dropped | wrong
        self.just_sorted &= ~fell & ~self.failed
        self.drop_count += dropped
        self.wrong_count += wrong
        self.box_lost, self.misplaced = dropped, wrong

        # a sorted, dropped or misplaced box is replaced by a new one on the pickup table
        self.reward_count = self.sorted_count.copy()            # the reward of this step still sees the old box
        self.sorted_count += self.just_sorted
        ids = np.nonzero(self.just_sorted | self.failed)[0]
        if ids.size:
            self._new_box(ids)
        time_out = self.episode_length >= self.max_episode_length
        return fell, time_out, {"dropped": dropped, "wrong_table": wrong, "sorted": self.just_sorted.copy()}

    def _task_terms(self, fell):
        cmd, sigma = self.commands, RewardCfg.tracking_sigma
        lin_err = (cmd[:, 0] - self.base_lin_vel[:, 0]) ** 2 + (cmd[:, 1] - self.base_lin_vel[:, 1]) ** 2
        rel_vel = self.box_vel - self.base_lin_vel_world
        return {
            "stage": NUM_STAGES * self.reward_count + self.stage + self.stage_progress,
            # the reward always knows where the robot should walk, whether the robot was told or not
            "tracking_lin_vel": np.exp(-lin_err / sigma),
            "tracking_ang_vel": np.exp(-(cmd[:, 2] - self.base_ang_vel[:, 2]) ** 2 / RewardCfg.tracking_sigma_yaw),
            "heading": 0.5 * (1.0 + np.cos(self.face_err)),
            "box_vel": (rel_vel ** 2).sum(axis=1) * self.gripped,
            "box_shove": self.box_shove,
            "sorted": self.just_sorted.astype(np.float64),
            "wrong_table": self.misplaced.astype(np.float64),
            "dropped": self.box_lost.astype(np.float64),
        }

    def _heading_frame(self, v):
        """World-frame vectors (N, 2 or 3) -> the robot's heading frame (x forward, y left)."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        out = v.copy()
        out[:, 0] = c * v[:, 0] + s * v[:, 1]
        out[:, 1] = -s * v[:, 0] + c * v[:, 1]
        return out

    def _compute_obs(self):
        base = self.base_pos
        box_pos = self.state[:, self._box_qpos][:, 0:3]
        bx, by, _ = _quat_to_axes(self.state[:, self._box_qpos][:, 3:7])
        box_vel = self.state[:, self._box_qvel][:, 0:3]
        color = np.zeros((self.num_envs, 3))
        color[np.arange(self.num_envs), self.box_color] = 1.0
        # tables listed by colour: first the red one, then green, then blue
        order = 1 + np.argsort(self.table_color, axis=1)
        tables = [self._heading_frame(self.table_xy[order[:, k]] - base[:, 0:2]) for k in range(3)]
        scan = np.repeat(base[:, 2:3] - RobotCfg.nominal_height, self.num_scan, axis=1)     # the floor is flat
        return self._finish_obs(self._proprioception() + [
            self.commands * self.guided[:, None],
            np.clip(scan, -1.0, 1.0) * ObsCfg.height_scale,
            self._heading_frame(box_pos - base),
            self._heading_frame(bx), self._heading_frame(by),
            self._heading_frame(box_vel - self.base_lin_vel_world),
            self.box_half * 4.0,
            color,
            self._heading_frame(self.grip_pos[:, 0] - base), self._heading_frame(self.grip_pos[:, 1] - base),
            np.clip(self.palm_touch / 20.0, 0.0, 3.0),
            self._heading_frame(self.table_xy[0][None, :] - base[:, 0:2]),
            *tables,
            np.clip(self.sorted_count[:, None] / 4.0, 0.0, 2.0),
        ])

    # ---------------------------------------------------------- guidance
    def curriculum_state(self):
        return {"guide": self.guide_probability, "mass_level": self.mass_level}

    def load_curriculum(self, state, rng):
        self.guide_probability = float(state.get("guide", self.guide_probability))
        self.mass_level = int(state.get("mass_level", self.mass_level))

    def on_iteration(self, iteration, stats):
        """Add a heavier box / withdraw the guide a little whenever the robots do well enough."""
        cfg = SortCfg
        if iteration % cfg.guide_every:
            return
        if self.mass_level < self.box_masses.size and stats.get("stage", 0.0) >= cfg.mass_threshold:
            self.mass_level += 1
        elif self.guide_probability > 0.0 and stats.get("sorted", 0.0) >= cfg.guide_threshold:
            self.guide_probability = max(0.0, round(self.guide_probability - cfg.guide_step, 4))

    def _episode_stats(self, ids, info):
        stats = super()._episode_stats(ids, info)
        stats["guided"] = self.guide_probability
        stats["sorted"] = float(self.sorted_count[ids].mean())
        stats["stage"] = float(self.max_stage[ids].mean())
        stats["dropped"] = float(self.drop_count[ids].mean())
        stats["wrong_table"] = float(self.wrong_count[ids].mean())
        stats["max_box_mass"] = float(self.box_masses[self.mass_level - 1])
        return stats


# Quick self-test: random actions, prints the simulation throughput
if __name__ == "__main__":
    import time

    env = SortEnv(num_envs=1024, guide=1.0)
    obs = env.reset()
    print(f"obs {obs.shape}, actions {env.num_actions}, threads {env.num_threads}, "
          f"models {sum(len(v) for v in env.variants)}, geoms {env.scene_models[0].ngeom}")
    steps = 100
    start = time.time()
    for _ in range(steps):
        obs, rew, done, info = env.step(np.random.randn(env.num_envs, env.num_actions) * 0.5)
    rate = steps * env.num_envs / (time.time() - start)
    print(f"{rate:,.0f} env steps/s  (mean reward {rew.mean():.4f}, obs finite: {np.isfinite(obs).all()})")
    env.close()
