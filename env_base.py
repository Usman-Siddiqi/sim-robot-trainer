"""
env_base.py

Author: Usman Siddiqi

The machinery shared by both tasks (hiking and sorting): thousands of robots are
simulated at once. Physics for all robots is stepped by `mujoco.rollout`, a C++
thread pool that uses every CPU core, while observations and rewards are
computed for the whole batch with vectorised numpy. The API is the usual
vectorised-RL one:

    env = HikeEnv(num_envs=2048)                    # or SortEnv
    obs = env.reset()
    obs, reward, done, info = env.step(actions)     # all arrays have a leading (num_envs,) axis

Finished episodes are reset automatically inside step().

Every robot is simulated with its own MjModel, picked from a small set:

    scene    which world the robot is in (hiking: which trail tile)
    variant  a copy of that scene with different physics (friction, payload, motor
             strength, ...) - the domain randomisation that makes the policy work
             in conditions it was not trained in

Action (20): joint position offsets from the default pose, tracked by PD servos.
"""

import copy
import os

import mujoco
import numpy as np
from mujoco import rollout

from config import DomainRandCfg, GaitCfg, ObsCfg, RewardCfg, RobotCfg, SimCfg

# sensordata layout (see <sensor> in robot/humanoid.xml)
_TOUCH = slice(0, 2)
_FOOT_POS = slice(2, 8)
_FOOT_VEL = slice(8, 14)
_FOOT_UP = slice(14, 20)
_PALM_TOUCH = slice(20, 22)
_GRIP_POS = slice(22, 28)
_NUM_SENSOR = 28


def quat_rotate_inverse(q, v):
    """Rotate world-frame vectors v (N,3) into the body frame of quaternions q (N,4; w,x,y,z)."""
    w = q[:, 0:1]
    u = q[:, 1:4]
    return v * (2.0 * w * w - 1.0) - 2.0 * w * np.cross(u, v) + 2.0 * u * np.sum(u * v, axis=1, keepdims=True)


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class RobotEnv:
    """Base class. Subclasses build the scenes and define the task (commands, rewards, resets)."""

    episode_length_s = 20.0
    default_arm_pos = RobotCfg.default_arm_pos

    def __init__(self, num_envs, num_threads=None, seed=0, randomize=True, physics=None, push=None):
        """
        Args:
            num_envs: number of robots simulated in parallel
            num_threads: physics threads (default: all logical cores)
            seed: random seed
            randomize: training-style randomisation (physics variants, pushes, sensor noise).
                       False gives the nominal robot without disturbances (evaluation / demos).
            physics: fixed physical parameters, e.g. {"friction": 0.4, "payload": 5.0,
                     "motor_strength": 0.85}; anything not given is nominal
            push: override of the push settings, e.g. {"max_push_vel": 0.8, "push_interval_s": 3.0}
        """
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.dt = SimCfg.control_dt()
        self.max_episode_length = int(round(self.episode_length_s / self.dt))
        self.randomize = bool(randomize)
        self.physics = physics
        self.obs_noise = bool(randomize and DomainRandCfg.obs_noise)
        self.push_robots = bool(push) or bool(randomize and DomainRandCfg.push_robots)
        push = push or {}
        self.max_push_vel = push.get("max_push_vel", DomainRandCfg.max_push_vel)
        self.push_interval = int(push.get("push_interval_s", DomainRandCfg.push_interval_s) / self.dt)
        self.num_threads = int(num_threads or os.cpu_count() or 1)
        self.num_actions = len(RobotCfg.joint_names)
        self.common_step = 0

    # ---------------------------------------------------------------- set-up
    def _init_sim(self, scene_models, variants=None, variants_per_scene=1):
        """
        Called by the subclass once its scene models exist. `variants` is a list (one entry per
        scene) of lists of models; by default `variants_per_scene` random ones are made.
        """
        self.scene_models = scene_models
        m = scene_models[0]
        nu = self.num_actions
        if m.nu != nu or m.nsensordata != _NUM_SENSOR or m.na != 0:
            raise RuntimeError("robot/humanoid.xml does not match what env_base.py expects")
        for i, name in enumerate(RobotCfg.joint_names):
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) != name:
                raise RuntimeError(f"actuator {i} should be '{name}'")

        # physics variants of every scene
        if variants is not None:
            self.variants = variants
        elif self.physics:
            self.variants = [[self._make_variant(sm, **self.physics)] for sm in scene_models]
        elif self.randomize:
            self.variants = [[self._make_variant(sm, **self._random_physics()) for _ in range(variants_per_scene)]
                             for sm in scene_models]
        else:
            self.variants = [[sm] for sm in scene_models]
        n = self.num_envs
        self.env_scene = np.zeros(n, dtype=np.int64)
        self.env_variant = self.rng.integers(0, 1 << 30, n)
        self._models = [m] * n

        self._rollout = rollout.Rollout(nthread=self.num_threads)
        self._datas = [mujoco.MjData(m) for _ in range(self.num_threads)]
        self._scratch = mujoco.MjData(m)
        self._state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        nstate = mujoco.mj_stateSize(m, self._state_spec)
        assert nstate == 1 + m.nq + m.nv
        self._qpos = slice(1, 1 + m.nq)
        self._qvel = slice(1 + m.nq, 1 + m.nq + m.nv)
        self.nq, self.nv = m.nq, m.nv

        self.state = np.zeros((n, nstate))
        self._state_traj = np.zeros((n, SimCfg.decimation, nstate))
        self._sensor_traj = np.zeros((n, SimCfg.decimation, _NUM_SENSOR))
        self._ctrl = np.zeros((n, 1, nu))

        # ------------------------------------------------------ robot constants
        nleg = RobotCfg.num_leg_joints
        self.default_joint_pos = np.array(RobotCfg.default_leg_pos + list(self.default_arm_pos), dtype=np.float64)
        self.action_scale = np.array([RobotCfg.leg_action_scale] * nleg + [RobotCfg.arm_action_scale] * (nu - nleg))
        jnt_range = m.jnt_range[1:1 + nu].copy()            # joint 0 is the free joint
        self.joint_lower, self.joint_upper = jnt_range[:, 0], jnt_range[:, 1]
        mid = 0.5 * (self.joint_lower + self.joint_upper)
        half = 0.5 * (self.joint_upper - self.joint_lower) * RewardCfg.soft_joint_limit
        self.soft_lower, self.soft_upper = mid - half, mid + half
        self.kp = m.actuator_gainprm[:, 0].copy()
        self.kd = -m.actuator_biasprm[:, 2].copy()
        self.torque_limit = m.actuator_forcerange[:, 1].copy()
        self._hip_idx = np.array([0, 1, 6, 7])              # hip yaw / roll joints
        self._arm_idx = np.arange(nleg, nu)
        # terrain is sampled under each foot at these offsets along the heading (heel .. ahead of toe)
        self._foot_probe = np.array([-0.10, 0.0, 0.10, 0.20])

        # ------------------------------------------------------------- buffers
        self.episode_length = np.zeros(n, dtype=np.int64)
        self.actions = np.zeros((n, nu))
        self.last_actions = np.zeros((n, nu))
        self.last_joint_vel = np.zeros((n, nu))
        self.contact = np.zeros((n, 2), dtype=bool)
        self.foot_pos = np.zeros((n, 2, 3))
        self.foot_vel = np.zeros((n, 2, 3))
        self.foot_up = np.zeros((n, 2, 3))
        self.foot_up[:, :, 2] = 1.0
        self.palm_touch = np.zeros((n, 2))
        self.grip_pos = np.zeros((n, 2, 3))
        self.standing = np.zeros(n, dtype=bool)             # the robot should stand instead of walking
        self._gravity = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))

    def _init_rewards(self, weights):
        self.reward_weights = {k: v for k, v in weights.items() if v != 0.0}
        self.episode_sums = {k: np.zeros(self.num_envs) for k in self.reward_weights}

    def _make_variant(self, model, friction=None, payload=0.0, com_offset=(0.0, 0.0, 0.0), motor_strength=1.0,
                      **task):
        """Copy of a model with different physical parameters."""
        m = copy.copy(model)
        if friction is not None:
            # the static world and the feet (a contact uses the larger friction of the two geoms)
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 or
                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) in ("l_foot", "r_foot")]
            m.geom_friction[geoms, 0] = friction
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        scale = (m.body_mass[torso] + payload) / m.body_mass[torso]
        m.body_mass[torso] *= scale
        m.body_inertia[torso] *= scale
        m.body_ipos[torso] += np.asarray(com_offset)
        m.actuator_gainprm[:, 0] *= motor_strength      # kp
        m.actuator_biasprm[:, 1:3] *= motor_strength    # -kp, -kd
        m.actuator_forcerange[:] *= motor_strength
        self._task_variant(m, **task)
        mujoco.mj_setConst(m, mujoco.MjData(m))
        return m

    def _task_variant(self, m, **task):
        """Hook: task-specific physics changes of a variant."""

    def _random_physics(self):
        """Random robot / ground parameters within the DomainRandCfg ranges."""
        dr, rng = DomainRandCfg, self.rng
        return dict(friction=rng.uniform(*dr.friction), payload=rng.uniform(*dr.payload),
                    com_offset=rng.uniform(-dr.com_offset, dr.com_offset, 3),
                    motor_strength=rng.uniform(*dr.motor_strength))

    def _assign_models(self, ids):
        """Refresh the per-robot model list after env_scene / env_variant changed."""
        for i in ids:
            pool = self.variants[self.env_scene[i]]
            self._models[i] = pool[self.env_variant[i] % len(pool)]

    # ================================================================== API
    def reset(self):
        """Reset every robot and return the first observation."""
        ids = np.arange(self.num_envs)
        self._reset_envs(ids)
        self._refresh_sensors(ids)
        self._update_kinematics()
        self._update_task()
        return self._compute_obs()

    def step(self, actions):
        """
        Advance all robots by one control step (SimCfg.decimation physics steps).

        Returns:
            obs (N, num_obs) float32, reward (N,) float32, done (N,) bool, info dict.
            info["time_outs"] marks episodes that ended without a failure (timeout /
            reached the goal); PPO bootstraps the value for those.
            info["fell"] says whether an episode ended in a failure.
            info["episode"] holds logging statistics of the episodes that just ended.
        """
        self.actions = np.clip(np.asarray(actions, dtype=np.float64), -ObsCfg.clip_actions, ObsCfg.clip_actions)
        target = self.default_joint_pos + self.action_scale * self.actions
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
        self.palm_touch = self._sensor_traj[:, :, _PALM_TOUCH].mean(axis=1)
        self.grip_pos = sensors[:, _GRIP_POS].reshape(-1, 2, 3)

        self.episode_length += 1
        self.common_step += 1
        self._update_kinematics()
        self._update_task()

        fell, time_out, info = self._check_termination()
        fell |= ~np.isfinite(self.state).all(axis=1)
        time_out &= ~fell
        done = fell | time_out
        reward = self._compute_reward(target, fell)

        # ---- resets
        info.update(time_outs=time_out, fell=fell)
        ids = np.nonzero(done)[0]
        if ids.size:
            info["episode"] = self._episode_stats(ids, info)
            self._on_episode_end(ids, info)
            self._reset_envs(ids)
            self._refresh_sensors(ids)

        if self.push_robots and self.common_step % self.push_interval == 0:
            self._push()

        self.last_actions[:] = self.actions
        self.last_actions[ids] = 0.0
        self.actions[ids] = 0.0
        if ids.size or self.push_robots:
            self._update_kinematics()
            self._update_task()
        self.last_joint_vel[:] = self.joint_vel
        return self._compute_obs(), reward.astype(np.float32), done, info

    # what train.py prints next to the return: (label, key in the episode statistics)
    progress_keys = ()

    def curriculum_state(self):
        """Whatever a later training run needs to continue the curriculum (stored in checkpoints)."""
        return {}

    def load_curriculum(self, state, rng):
        pass

    def model_of(self, index):
        """The MjModel robot `index` is currently simulated with (use it for rendering)."""
        return self._models[index]

    def write_data(self, index, data):
        """Copy the state of robot `index` into an MjData (for rendering)."""
        model = self._models[index]
        mujoco.mj_setState(model, data, self.state[index], self._state_spec)
        mujoco.mj_forward(model, data)

    def close(self):
        self._rollout.close()

    # ============================================================ task hooks
    def _reset_envs(self, ids):
        raise NotImplementedError

    def _update_task(self):
        """Commands / task state, called after every kinematics update."""

    def _check_termination(self):
        """-> fell (N,) bool, time_out (N,) bool, info dict."""
        raise NotImplementedError

    def _task_terms(self, fell):
        """-> dict of task reward terms."""
        raise NotImplementedError

    def _on_episode_end(self, ids, info):
        pass

    def _episode_stats(self, ids, info):
        stats = {f"reward/{k}": float(v[ids].mean()) for k, v in self.episode_sums.items()}
        stats["episode_length"] = float(self.episode_length[ids].mean())
        stats["fell"] = float(info["fell"][ids].mean())
        return stats

    def _ground(self, x, y):
        """Terrain height under world points; x, y have shape (N, ...)."""
        return np.zeros(np.broadcast(x, y).shape)

    # ============================================================ internals
    def _update_kinematics(self):
        qpos = self.state[:, self._qpos]
        qvel = self.state[:, self._qvel]
        nu = self.num_actions
        self.base_pos = qpos[:, 0:3]
        self.base_quat = qpos[:, 3:7]
        self.joint_pos = qpos[:, 7:7 + nu]
        self.joint_vel = qvel[:, 6:6 + nu]
        self.base_lin_vel_world = qvel[:, 0:3]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, qvel[:, 0:3])
        self.base_ang_vel = qvel[:, 3:6]                    # MuJoCo free joints: already in the body frame
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self._gravity)
        w, x, y, z = self.base_quat.T
        self.yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _robot_reset_state(self, ids, x, y, z, yaw):
        """Write a fresh robot (default pose + noise) at the given places into the state buffer."""
        n, nu, rng = ids.size, self.num_actions, self.rng
        qpos = np.zeros((n, 7 + nu))
        qpos[:, 0], qpos[:, 1], qpos[:, 2] = x, y, z
        qpos[:, 3] = np.cos(yaw / 2)
        qpos[:, 6] = np.sin(yaw / 2)
        noise = DomainRandCfg.init_joint_noise
        qpos[:, 7:] = np.clip(self.default_joint_pos + rng.uniform(-noise, noise, (n, nu)),
                              self.joint_lower, self.joint_upper)
        qvel = np.zeros((n, 6 + nu))
        qvel[:, 0:6] = rng.uniform(-DomainRandCfg.init_vel_noise, DomainRandCfg.init_vel_noise, (n, 6))
        self.state[ids, 0] = 0.0
        self.state[ids, 1:8 + nu] = qpos
        self.state[ids, 1 + self.nq:1 + self.nq + 6 + nu] = qvel
        self.episode_length[ids] = 0
        self.last_actions[ids] = 0.0
        self.last_joint_vel[ids] = 0.0
        self.contact[ids] = True
        for v in self.episode_sums.values():
            v[ids] = 0.0

    def _refresh_sensors(self, ids):
        """Foot and palm positions of robots that were just reset (otherwise they would be one step stale)."""
        d = self._scratch
        for i in ids:
            model = self._models[i]
            mujoco.mj_setState(model, d, self.state[i], self._state_spec)
            mujoco.mj_kinematics(model, d)
            mujoco.mj_sensorPos(model, d)
            self.foot_pos[i] = d.sensordata[_FOOT_POS].reshape(2, 3)
            self.foot_up[i] = d.sensordata[_FOOT_UP].reshape(2, 3)
            self.grip_pos[i] = d.sensordata[_GRIP_POS].reshape(2, 3)
        self.foot_vel[ids] = 0.0
        self.palm_touch[ids] = 0.0

    def _push(self):
        """Random horizontal shove so the policy learns to recover its balance."""
        v = self.max_push_vel
        self.state[:, self._qvel][:, 0:2] += self.rng.uniform(-v, v, (self.num_envs, 2))

    def _leg_phase(self):
        """Gait phase in [0, 1) of the left and right leg (half a cycle apart) -> (N, 2)."""
        phase = (self.episode_length * self.dt / GaitCfg.period) % 1.0
        return np.stack([phase, (phase + 0.5) % 1.0], axis=1)

    def _foot_terrain(self):
        """Highest terrain point under / just ahead of each foot -> (N, 2)."""
        c, s = np.cos(self.yaw)[:, None, None], np.sin(self.yaw)[:, None, None]
        px = self.foot_pos[:, :, 0:1] + c * self._foot_probe
        py = self.foot_pos[:, :, 1:2] + s * self._foot_probe
        return self._ground(px, py).max(axis=2)

    def _common_terms(self, target, fell):
        """Reward terms shared by all tasks: gait, posture, feet, effort."""
        cfg = RewardCfg
        terms = {"alive": 1.0 - fell}

        # --- gait: each foot should be on the ground during its half of the gait cycle
        stance = (self._leg_phase() < GaitCfg.stance_fraction) | self.standing[:, None]
        terms["gait_contact"] = (self.contact == stance).sum(axis=1)
        self.foot_ground = self._foot_terrain()
        clearance = self.foot_pos[:, :, 2] - self.foot_ground
        terms["swing_height"] = (((clearance - cfg.swing_height_target) ** 2) * ~self.contact).sum(axis=1) \
            * ~self.standing

        # --- posture
        terms["lin_vel_z"] = self.base_lin_vel_world[:, 2] ** 2
        terms["ang_vel_xy"] = (self.base_ang_vel[:, 0:2] ** 2).sum(axis=1)
        terms["orientation"] = (self.projected_gravity[:, 0:2] ** 2).sum(axis=1)
        base_height = self.base_pos[:, 2] - self.foot_ground.mean(axis=1)
        terms["base_height"] = np.clip(RobotCfg.nominal_height - 0.06 - base_height, 0.0, None) ** 2
        terms["hip_pos"] = ((self.joint_pos[:, self._hip_idx] - self.default_joint_pos[self._hip_idx]) ** 2).sum(axis=1)
        terms["arm_pos"] = ((self.joint_pos[:, self._arm_idx] - self.default_joint_pos[self._arm_idx]) ** 2).sum(axis=1)

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
        return terms

    def _compute_reward(self, target, fell):
        terms = self._common_terms(target, fell)
        terms.update(self._task_terms(fell))
        reward = np.zeros(self.num_envs)
        for name, weight in self.reward_weights.items():
            value = np.nan_to_num(terms[name] * (weight * self.dt), nan=0.0, posinf=0.0, neginf=0.0)
            reward += value
            self.episode_sums[name] += value
        return reward

    def _proprioception(self):
        """The part of the observation both tasks share (3 + 3 + 3 + 20 + 20 + 20 + 2 = 71 numbers)."""
        phase = 2.0 * np.pi * self._leg_phase()[:, 0:1]
        return [
            self.base_lin_vel * ObsCfg.lin_vel_scale,
            self.base_ang_vel * ObsCfg.ang_vel_scale,
            self.projected_gravity,
            self.joint_pos - self.default_joint_pos,
            self.joint_vel * ObsCfg.joint_vel_scale,
            self.actions,
            np.sin(phase), np.cos(phase),
        ]

    def _proprioception_noise(self):
        dr, nu = DomainRandCfg, self.num_actions
        return [
            np.full(3, dr.noise_lin_vel * ObsCfg.lin_vel_scale), np.full(3, dr.noise_ang_vel * ObsCfg.ang_vel_scale),
            np.full(3, dr.noise_gravity), np.full(nu, dr.noise_joint_pos),
            np.full(nu, dr.noise_joint_vel * ObsCfg.joint_vel_scale), np.zeros(nu), np.zeros(2),
        ]

    def _finish_obs(self, parts):
        obs = np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.obs_noise:
            obs += self._noise_scale * self.rng.uniform(-1.0, 1.0, obs.shape).astype(np.float32)
        return obs
