"""
environment.py

Author: Usman Siddiqi

Batched MuJoCo environment for the biped: thousands of robots are simulated at
once. Physics for all robots is stepped by `mujoco.rollout`, a C++ thread pool
that uses every CPU core, while observations and rewards are computed for the
whole batch with vectorised numpy. The API is the usual vectorised-RL one:

    env = BipedEnv(num_envs=2048)
    obs = env.reset()
    obs, reward, done, info = env.step(actions)     # all arrays have a leading (num_envs,) axis

Task: follow a commanded forward speed and heading over flat ground, rough
ground, staircases and slopes. Finished episodes are reset automatically inside
step(), and a curriculum moves each robot to harder / easier tiles depending on
how it did.

So that the policy also works in conditions it was not trained in, training
randomises the physics (friction, payload, motor strength), shoves the robots
around and adds sensor noise - see DomainRandCfg.

Action  (12): joint position offsets from the default pose, tracked by PD servos.
Observation : base velocity, angular velocity, gravity direction, command,
              joint positions / velocities, previous action, gait clock and a
              height scan of the terrain around the robot.
"""

import copy
import os

import mujoco
import numpy as np
from mujoco import rollout

from config import CommandCfg, DomainRandCfg, ObsCfg, RewardCfg, RobotCfg, SimCfg, TerrainCfg
from terrain import Terrain, build_model

# sensordata layout (see <sensor> in robot/biped.xml)
_TOUCH = slice(0, 2)
_FOOT_POS = slice(2, 8)
_FOOT_VEL = slice(8, 14)
_FOOT_UP = slice(14, 20)
_NUM_SENSOR = 20


def quat_rotate_inverse(q, v):
    """Rotate world-frame vectors v (N,3) into the body frame of quaternions q (N,4; w,x,y,z)."""
    w = q[:, 0:1]
    u = q[:, 1:4]
    return v * (2.0 * w * w - 1.0) - 2.0 * w * np.cross(u, v) + 2.0 * u * np.sum(u * v, axis=1, keepdims=True)


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class BipedEnv:
    def __init__(self, num_envs, flat_only=False, num_threads=None, seed=0, terrain_rows=None,
                 terrain_seed=0, curriculum=None, fixed_level=None, randomize=True, physics=None,
                 push=None, fixed_speed=None, heading_offset=None, resample_commands=True):
        """
        Args:
            num_envs: number of robots simulated in parallel
            flat_only: flat ground only (no obstacles)
            num_threads: physics threads (default: all logical cores)
            seed: random seed
            terrain_rows: terrain rows to build (default: TerrainCfg.rows, the training terrains)
            terrain_seed: seed of the random parts of the terrain
            curriculum: override TerrainCfg.curriculum (False keeps levels fixed)
            fixed_level: start every robot on this level instead of the easy ones
            randomize: training-style randomisation (physics variants, pushes, sensor noise).
                       False gives the nominal robot without disturbances (evaluation / demos).
            physics: fixed physical parameters, e.g. {"friction": 0.4, "payload": 5.0,
                     "motor_strength": 0.85}; anything not given is nominal
            push: override of the push settings, e.g. {"max_push_vel": 0.8, "push_interval_s": 3.0}
            fixed_speed: always command this forward speed [m/s]
            heading_offset: approach obstacles this far from square-on [rad] instead of randomly
            resample_commands: pick a new random speed / heading every few seconds
        """
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.dt = SimCfg.control_dt()
        self.max_episode_length = int(round(SimCfg.episode_length_s / self.dt))
        self.curriculum = TerrainCfg.curriculum if curriculum is None else curriculum
        self.fixed_speed = fixed_speed
        self.heading_offset = heading_offset
        self.resample_commands = resample_commands
        self.obs_noise = bool(randomize and DomainRandCfg.obs_noise)
        self.push_robots = bool(push) or bool(randomize and DomainRandCfg.push_robots)
        push = push or {}
        self.max_push_vel = push.get("max_push_vel", DomainRandCfg.max_push_vel)
        self.push_interval = int(push.get("push_interval_s", DomainRandCfg.push_interval_s) / self.dt)

        # ------------------------------------------------------------- model
        rows = TerrainCfg.flat_rows if flat_only else (terrain_rows or TerrainCfg.rows)
        self.terrain = Terrain(rows, TerrainCfg, seed=terrain_seed)
        self.model = build_model(SimCfg, self.terrain)       # the nominal model (also used for rendering)
        m = self.model
        self.num_actions = m.nu
        if m.nq != 7 + m.nu or m.nv != 6 + m.nu or m.na != 0 or m.nsensordata != _NUM_SENSOR:
            raise RuntimeError("robot/biped.xml does not match what environment.py expects")
        for i, name in enumerate(RobotCfg.joint_names):
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) != name:
                raise RuntimeError(f"actuator {i} should be '{name}'")

        # physics variants: robot i is simulated with variant i % num_variants
        if physics:
            self.variants = [self._make_variant(**physics)]
        elif randomize:
            self.variants = [self._random_variant() for _ in range(DomainRandCfg.num_variants)]
        else:
            self.variants = [m]
        self._models = [self.variants[i % len(self.variants)] for i in range(self.num_envs)]

        self.num_threads = int(num_threads or os.cpu_count() or 1)
        self._rollout = rollout.Rollout(nthread=self.num_threads)
        self._datas = [mujoco.MjData(m) for _ in range(self.num_threads)]
        self._state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        nstate = mujoco.mj_stateSize(m, self._state_spec)
        assert nstate == 1 + m.nq + m.nv
        self._qpos = slice(1, 1 + m.nq)
        self._qvel = slice(1 + m.nq, 1 + m.nq + m.nv)

        n, nu = self.num_envs, self.num_actions
        self.state = np.zeros((n, nstate))
        self._state_traj = np.zeros((n, SimCfg.decimation, nstate))
        self._sensor_traj = np.zeros((n, SimCfg.decimation, _NUM_SENSOR))
        self._ctrl = np.zeros((n, 1, nu))

        # ------------------------------------------------------ robot constants
        self.default_joint_pos = np.array(RobotCfg.default_joint_pos, dtype=np.float64)
        jnt_range = m.jnt_range[1:1 + nu].copy()            # joint 0 is the free joint
        self.joint_lower, self.joint_upper = jnt_range[:, 0], jnt_range[:, 1]
        mid = 0.5 * (self.joint_lower + self.joint_upper)
        half = 0.5 * (self.joint_upper - self.joint_lower) * RewardCfg.soft_joint_limit
        self.soft_lower, self.soft_upper = mid - half, mid + half
        self.kp = m.actuator_gainprm[:, 0].copy()
        self.kd = -m.actuator_biasprm[:, 2].copy()
        self.torque_limit = m.actuator_forcerange[:, 1].copy()
        self._hip_idx = np.array([0, 1, 6, 7])              # hip yaw / roll joints

        # ---------------------------------------------------------- observation
        sx, sy = np.meshgrid(np.array(ObsCfg.scan_x), np.array(ObsCfg.scan_y), indexing="ij")
        self._scan_x, self._scan_y = sx.ravel(), sy.ravel()
        self.num_scan = self._scan_x.size
        self.num_obs = 3 + 3 + 3 + 3 + nu + nu + nu + 2 + self.num_scan
        # terrain is sampled under each foot at these offsets along the heading (heel .. ahead of toe)
        self._foot_probe = np.array([-0.10, 0.0, 0.10, 0.20])

        # ------------------------------------------------------------- buffers
        self.episode_length = np.zeros(n, dtype=np.int64)
        self.actions = np.zeros((n, nu))
        self.last_actions = np.zeros((n, nu))
        self.last_joint_vel = np.zeros((n, nu))
        self.commands = np.zeros((n, 3))                    # vx, vy, yaw rate
        self.heading_target = np.zeros(n)
        self.contact = np.zeros((n, 2), dtype=bool)
        self.foot_pos = np.zeros((n, 2, 3))
        self.foot_vel = np.zeros((n, 2, 3))
        self.foot_up = np.zeros((n, 2, 3))
        self.foot_up[:, :, 2] = 1.0
        self.common_step = 0

        # terrain assignment: each robot has a fixed terrain row and a level that changes
        weights = np.array([row.get("weight", 1.0) for row in self.terrain.rows], dtype=np.float64)
        if weights.sum() <= 0:
            weights = (~self.terrain.row_is_flat).astype(np.float64)
        bounds = np.round(np.cumsum(weights / weights.sum()) * n).astype(np.int64)
        self.env_row = np.searchsorted(bounds, np.arange(n), side="right").clip(0, self.terrain.num_rows - 1)
        self.is_flat = self.terrain.row_is_flat[self.env_row]
        max_level = self.terrain.num_levels - 1
        if fixed_level is not None:
            self.env_level = np.full(n, int(np.clip(fixed_level, 0, max_level)), dtype=np.int64)
        else:
            self.env_level = self.rng.integers(0, min(TerrainCfg.max_init_level, max_level) + 1, size=n)

        # sensor noise amplitude for every entry of the observation
        dr = DomainRandCfg
        self._noise_scale = np.concatenate([
            np.full(3, dr.noise_lin_vel * ObsCfg.lin_vel_scale), np.full(3, dr.noise_ang_vel * ObsCfg.ang_vel_scale),
            np.full(3, dr.noise_gravity), np.zeros(3), np.full(nu, dr.noise_joint_pos),
            np.full(nu, dr.noise_joint_vel * ObsCfg.joint_vel_scale), np.zeros(nu), np.zeros(2),
            np.full(self.num_scan, dr.noise_height_scan * ObsCfg.height_scale),
        ]).astype(np.float32)

        self.reward_weights = {k: v for k, v in RewardCfg.weights.items() if v != 0.0}
        self.episode_sums = {k: np.zeros(n) for k in self.reward_weights}
        self._gravity = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))

    # ================================================================== API
    def reset(self):
        """Reset every robot and return the first observation."""
        self._reset_envs(np.arange(self.num_envs))
        self._update_kinematics()
        self._update_commands()
        return self._compute_obs()

    def step(self, actions):
        """
        Advance all robots by one control step (SimCfg.decimation physics steps).

        Returns:
            obs (N, num_obs) float32, reward (N,) float32, done (N,) bool, info dict.
            info["time_outs"] marks episodes that ended without the robot falling
            (timeout / reached the goal); PPO bootstraps the value for those.
            info["fell"] / info["reached_goal"] say why an episode ended.
            info["episode"] holds logging statistics of the episodes that just ended.
        """
        self.actions = np.clip(np.asarray(actions, dtype=np.float64), -ObsCfg.clip_actions, ObsCfg.clip_actions)
        target = self.default_joint_pos + RobotCfg.action_scale * self.actions
        np.clip(target, self.joint_lower, self.joint_upper, out=target)
        self._ctrl[:, 0, :] = target

        # ---- physics: all robots, all cores
        self._rollout.rollout(
            self._models, self._datas, self.state, self._ctrl, nstep=SimCfg.decimation,
            state=self._state_traj, sensordata=self._sensor_traj,
        )
        self.state[:] = self._state_traj[:, -1]
        sensors = self._sensor_traj[:, -1]
        # a foot counts as "in contact" if it touched the ground during any physics sub-step
        self.contact = self._sensor_traj[:, :, _TOUCH].max(axis=1) > 1.0
        self.foot_pos = sensors[:, _FOOT_POS].reshape(-1, 2, 3)
        self.foot_vel = sensors[:, _FOOT_VEL].reshape(-1, 2, 3)
        self.foot_up = sensors[:, _FOOT_UP].reshape(-1, 2, 3)

        self.episode_length += 1
        self.common_step += 1
        self._update_kinematics()
        self._update_commands()

        # ---- termination
        ground = self.terrain.height(self.base_pos[:, 0], self.base_pos[:, 1])
        fell = (self.base_pos[:, 2] - ground < RobotCfg.min_height) | \
               (self.projected_gravity[:, 2] > RobotCfg.max_tilt_gravity_z)
        fell |= ~np.isfinite(self.state).all(axis=1)
        dist = self.terrain.distance_from_center(self.base_pos[:, 0], self.base_pos[:, 1], self.env_row, self.env_level)
        reached_goal = ~self.is_flat & (dist > TerrainCfg.success_distance)
        left_flat = self.is_flat & (self.terrain.num_rows > 1) & (self.base_pos[:, 1] > self.terrain.tile - 1.0)
        time_out = (self.episode_length >= self.max_episode_length) | reached_goal | left_flat
        time_out &= ~fell
        done = fell | time_out

        reward = self._compute_reward(target, fell)

        # ---- resets (+ curriculum)
        info = {"time_outs": time_out, "fell": fell, "reached_goal": reached_goal}
        ids = np.nonzero(done)[0]
        if ids.size:
            info["episode"] = self._episode_stats(ids, fell, reached_goal)
            self._update_curriculum(ids, fell, reached_goal, dist)
            self._reset_envs(ids)

        if self.push_robots and self.common_step % self.push_interval == 0:
            self._push()

        self.last_actions[:] = self.actions
        self.last_actions[ids] = 0.0
        self.actions[ids] = 0.0
        if ids.size or self.push_robots:
            self._update_kinematics()
            self._update_commands()
        self.last_joint_vel[:] = self.joint_vel
        return self._compute_obs(), reward.astype(np.float32), done, info

    def write_data(self, index, data):
        """Copy the state of robot `index` into an MjData (for rendering)."""
        mujoco.mj_setState(self.model, data, self.state[index], self._state_spec)
        mujoco.mj_forward(self.model, data)

    def close(self):
        self._rollout.close()

    # ============================================================ internals
    def _make_variant(self, friction=None, payload=0.0, com_offset=(0.0, 0.0, 0.0), motor_strength=1.0):
        """Copy of the nominal model with different physical parameters."""
        m = copy.copy(self.model)
        if friction is not None:
            m.geom_friction[:, 0] = friction            # feet and ground: the contact uses the larger of the two
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        scale = (m.body_mass[torso] + payload) / m.body_mass[torso]
        m.body_mass[torso] *= scale
        m.body_inertia[torso] *= scale
        m.body_ipos[torso] += np.asarray(com_offset)
        m.actuator_gainprm[:, 0] *= motor_strength      # kp
        m.actuator_biasprm[:, 1:3] *= motor_strength    # -kp, -kd
        m.actuator_forcerange[:] *= motor_strength
        mujoco.mj_setConst(m, mujoco.MjData(m))
        return m

    def _random_variant(self):
        dr, rng = DomainRandCfg, self.rng
        return self._make_variant(
            friction=rng.uniform(*dr.friction), payload=rng.uniform(*dr.payload),
            com_offset=rng.uniform(-dr.com_offset, dr.com_offset, 3),
            motor_strength=rng.uniform(*dr.motor_strength),
        )

    def _update_kinematics(self):
        qpos = self.state[:, self._qpos]
        qvel = self.state[:, self._qvel]
        self.base_pos = qpos[:, 0:3]
        self.base_quat = qpos[:, 3:7]
        self.joint_pos = qpos[:, 7:]
        self.joint_vel = qvel[:, 6:]
        self.base_lin_vel_world = qvel[:, 0:3]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, qvel[:, 0:3])
        self.base_ang_vel = qvel[:, 3:6]                    # MuJoCo free joints: already in the body frame
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self._gravity)
        w, x, y, z = self.base_quat.T
        self.yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _update_commands(self):
        """Resample speeds / headings when due and turn the heading error into a yaw-rate command."""
        due = (self.episode_length % int(CommandCfg.resample_time_s / self.dt) == 0) & (self.episode_length > 0)
        ids = np.nonzero(due)[0]
        if ids.size and self.resample_commands:
            self._sample_speed(ids)
            flat = ids[self.is_flat[ids]]
            self.heading_target[flat] = self.rng.uniform(-np.pi, np.pi, flat.size)
        err = wrap_to_pi(self.heading_target - self.yaw)
        self.commands[:, 2] = np.clip(CommandCfg.heading_gain * err, -CommandCfg.max_yaw_rate, CommandCfg.max_yaw_rate)

    def _sample_speed(self, ids):
        if self.fixed_speed is not None:
            self.commands[ids, 0] = self.fixed_speed
        else:
            lo, hi = CommandCfg.lin_vel_x
            self.commands[ids, 0] = self.rng.uniform(lo, hi, ids.size)

    def _reset_envs(self, ids):
        n = ids.size
        m = self.model
        rng = self.rng
        cx, cy = self.terrain.tile_center(self.env_row[ids], self.env_level[ids])
        x = cx + rng.uniform(-0.4, 0.4, n)
        y = cy + rng.uniform(-0.4, 0.4, n)

        # heading: anywhere on flat ground, towards one of the four sides of an obstacle tile
        flat = self.is_flat[ids]
        if self.heading_offset is None:
            offset = rng.uniform(-CommandCfg.heading_noise, CommandCfg.heading_noise, n)
        else:
            offset = self.heading_offset * rng.choice([-1.0, 1.0], n)
        heading = np.where(flat, rng.uniform(-np.pi, np.pi, n), rng.integers(0, 4, n) * (np.pi / 2) + offset)
        yaw = np.where(flat, rng.uniform(-np.pi, np.pi, n), heading + rng.uniform(-0.3, 0.3, n))
        self.heading_target[ids] = wrap_to_pi(heading)

        qpos = np.zeros((n, m.nq))
        qpos[:, 0] = x
        qpos[:, 1] = y
        qpos[:, 2] = self.terrain.height(x, y) + RobotCfg.init_height
        qpos[:, 3] = np.cos(yaw / 2)
        qpos[:, 6] = np.sin(yaw / 2)
        noise = DomainRandCfg.init_joint_noise
        qpos[:, 7:] = np.clip(self.default_joint_pos + rng.uniform(-noise, noise, (n, m.nu)),
                              self.joint_lower, self.joint_upper)
        qvel = np.zeros((n, m.nv))
        qvel[:, 0:6] = rng.uniform(-DomainRandCfg.init_vel_noise, DomainRandCfg.init_vel_noise, (n, 6))

        self.state[ids, 0] = 0.0
        self.state[ids, self._qpos] = qpos
        self.state[ids, self._qvel] = qvel

        self._sample_speed(ids)
        self.episode_length[ids] = 0
        self.last_actions[ids] = 0.0
        self.last_joint_vel[ids] = 0.0
        self.contact[ids] = True
        for v in self.episode_sums.values():
            v[ids] = 0.0

    def _push(self):
        """Random horizontal shove so the policy learns to recover its balance."""
        v = self.max_push_vel
        self.state[:, self._qvel][:, 0:2] += self.rng.uniform(-v, v, (self.num_envs, 2))

    def _update_curriculum(self, ids, fell, reached_goal, dist):
        """Obstacle robots: crossed the whole tile -> harder level, fell or got stuck -> easier."""
        if not self.curriculum or self.terrain.num_levels == 1:
            return
        ids = ids[~self.is_flat[ids]]
        up = reached_goal[ids]
        down = (fell[ids] | (dist[ids] < 0.5 * TerrainCfg.success_distance)) & ~up
        level = self.env_level[ids] + up.astype(np.int64) - down.astype(np.int64)
        # robots that beat the hardest level are sent to a random level to keep practising all of them
        solved = level >= self.terrain.num_levels
        level[solved] = self.rng.integers(0, self.terrain.num_levels, int(solved.sum()))
        self.env_level[ids] = np.clip(level, 0, None)

    def _foot_terrain(self):
        """Highest terrain point under / just ahead of each foot -> (N, 2)."""
        c, s = np.cos(self.yaw)[:, None, None], np.sin(self.yaw)[:, None, None]
        px = self.foot_pos[:, :, 0:1] + c * self._foot_probe
        py = self.foot_pos[:, :, 1:2] + s * self._foot_probe
        return self.terrain.height(px, py).max(axis=2)

    def _compute_reward(self, target, fell):
        cfg = RewardCfg
        cmd = self.commands
        terms = {}

        # --- task
        lin_err = (cmd[:, 0] - self.base_lin_vel[:, 0]) ** 2 + (cmd[:, 1] - self.base_lin_vel[:, 1]) ** 2
        terms["tracking_lin_vel"] = np.exp(-lin_err / cfg.tracking_sigma)
        terms["tracking_ang_vel"] = np.exp(-(cmd[:, 2] - self.base_ang_vel[:, 2]) ** 2 / cfg.tracking_sigma)
        terms["alive"] = 1.0 - fell

        # --- gait: each foot should be on the ground during its half of the gait cycle
        stance = self._leg_phase() < CommandCfg.stance_fraction
        terms["gait_contact"] = (self.contact == stance).sum(axis=1)
        foot_ground = self._foot_terrain()
        clearance = self.foot_pos[:, :, 2] - foot_ground
        terms["swing_height"] = (((clearance - cfg.swing_height_target) ** 2) * ~self.contact).sum(axis=1)

        # --- posture
        terms["lin_vel_z"] = self.base_lin_vel_world[:, 2] ** 2
        terms["ang_vel_xy"] = (self.base_ang_vel[:, 0:2] ** 2).sum(axis=1)
        terms["orientation"] = (self.projected_gravity[:, 0:2] ** 2).sum(axis=1)
        base_height = self.base_pos[:, 2] - foot_ground.mean(axis=1)
        terms["base_height"] = np.clip(RobotCfg.nominal_height - 0.06 - base_height, 0.0, None) ** 2
        terms["hip_pos"] = ((self.joint_pos[:, self._hip_idx] - self.default_joint_pos[self._hip_idx]) ** 2).sum(axis=1)

        # --- feet
        d = self.foot_pos[:, 0, 0:2] - self.foot_pos[:, 1, 0:2]                  # left - right
        lateral = -np.sin(self.yaw) * d[:, 0] + np.cos(self.yaw) * d[:, 1]
        terms["feet_distance"] = np.clip(cfg.min_feet_distance - lateral, 0.0, None)
        terms["foot_flat"] = (self.foot_up[:, :, 0:2] ** 2).sum(axis=(1, 2))
        terms["feet_slip"] = ((self.foot_vel[:, :, 0:2] ** 2).sum(axis=2) * self.contact).sum(axis=1)

        # --- effort / smoothness
        torque = np.clip(self.kp * (target - self.joint_pos) - self.kd * self.joint_vel,
                         -self.torque_limit, self.torque_limit)
        terms["torques"] = (torque ** 2).sum(axis=1)
        terms["joint_vel"] = (self.joint_vel ** 2).sum(axis=1)
        terms["joint_acc"] = (((self.joint_vel - self.last_joint_vel) / self.dt) ** 2).sum(axis=1)
        terms["action_rate"] = ((self.actions - self.last_actions) ** 2).sum(axis=1)
        over = np.clip(self.soft_lower - self.joint_pos, 0.0, None) + np.clip(self.joint_pos - self.soft_upper, 0.0, None)
        terms["joint_limits"] = over.sum(axis=1)
        terms["termination"] = fell.astype(np.float64)

        reward = np.zeros(self.num_envs)
        for name, weight in self.reward_weights.items():
            value = np.nan_to_num(terms[name] * (weight * self.dt), nan=0.0, posinf=0.0, neginf=0.0)
            reward += value
            self.episode_sums[name] += value
        return reward

    def _leg_phase(self):
        """Gait phase in [0, 1) of the left and right leg (half a cycle apart) -> (N, 2)."""
        phase = (self.episode_length * self.dt / CommandCfg.gait_period) % 1.0
        return np.stack([phase, (phase + 0.5) % 1.0], axis=1)

    def _compute_obs(self):
        c, s = np.cos(self.yaw)[:, None], np.sin(self.yaw)[:, None]
        px = self.base_pos[:, 0:1] + c * self._scan_x - s * self._scan_y
        py = self.base_pos[:, 1:2] + s * self._scan_x + c * self._scan_y
        scan = self.base_pos[:, 2:3] - RobotCfg.nominal_height - self.terrain.height(px, py)
        phase = 2.0 * np.pi * self._leg_phase()[:, 0:1]
        obs = np.concatenate([
            self.base_lin_vel * ObsCfg.lin_vel_scale,
            self.base_ang_vel * ObsCfg.ang_vel_scale,
            self.projected_gravity,
            self.commands,
            self.joint_pos - self.default_joint_pos,
            self.joint_vel * ObsCfg.joint_vel_scale,
            self.actions,
            np.sin(phase), np.cos(phase),
            np.clip(scan, -1.0, 1.0) * ObsCfg.height_scale,
        ], axis=1)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.obs_noise:
            obs += self._noise_scale * self.rng.uniform(-1.0, 1.0, obs.shape).astype(np.float32)
        return obs

    def _episode_stats(self, ids, fell, reached_goal):
        stats = {f"reward/{k}": float(v[ids].mean()) for k, v in self.episode_sums.items()}
        stats["episode_length"] = float(self.episode_length[ids].mean())
        stats["fell"] = float(fell[ids].mean())
        obstacle = ids[~self.is_flat[ids]]
        if obstacle.size:
            stats["success"] = float(reached_goal[obstacle].mean())
        if not self.is_flat.all():
            stats["terrain_level"] = float(self.env_level[~self.is_flat].mean())
            for row in np.unique(self.env_row[~self.is_flat]):
                stats[f"level/{row}_{self.terrain.row_names[row]}"] = float(self.env_level[self.env_row == row].mean())
        return stats


# Quick self-test: random actions, prints the simulation throughput
if __name__ == "__main__":
    import time

    env = BipedEnv(num_envs=1024)
    obs = env.reset()
    print(f"obs {obs.shape}, actions {env.num_actions}, threads {env.num_threads}, "
          f"geoms {env.model.ngeom}")
    steps = 100
    start = time.time()
    for _ in range(steps):
        obs, rew, done, info = env.step(np.random.randn(env.num_envs, env.num_actions) * 0.5)
    rate = steps * env.num_envs / (time.time() - start)
    print(f"{rate:,.0f} env steps/s  (mean reward {rew.mean():.4f}, obs finite: {np.isfinite(obs).all()})")
    env.close()
