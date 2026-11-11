"""
env_hike.py

Author: Usman Siddiqi

Task 1 - hiking. The robot follows a winding mountain trail (see terrain.py):
rock steps of different heights, stepping stones with gaps, narrow ridges,
boulders to get around. A small navigation layer turns "where does the trail go"
into a velocity command (forward speed, sideways speed, turning rate); the
policy has to realise that command with its 20 joints while looking at a height
scan of the ground ahead, and its arms are free to swing for balance.

Leaving the trail, falling into a gap or falling over ends the episode. A
curriculum moves each robot to harder / easier trails depending on how it did.

Observation (218): base velocity, angular velocity, gravity direction, joint
positions / velocities, previous action, gait clock, velocity command and a
16 x 9 height scan.
"""

import numpy as np

from config import DomainRandCfg, EvalCfg, GaitCfg, HikeCfg, ObsCfg, RewardCfg, RobotCfg, SimCfg
from env_base import RobotEnv, wrap_to_pi
from terrain import Terrain


class HikeEnv(RobotEnv):
    task = "hike"
    episode_length_s = HikeCfg.episode_length_s

    def __init__(self, num_envs, flat_only=False, num_threads=None, seed=0, rows=None, terrain_seed=0,
                 curriculum=None, fixed_level=None, randomize=True, physics=None, push=None,
                 fixed_speed=None, resample_commands=True):
        """
        Args (see RobotEnv for the rest):
            flat_only: flat ground only (no trails)
            rows: terrain rows to build (default: HikeCfg.rows, the training trails)
            terrain_seed: seed of the random trail layouts
            curriculum: override HikeCfg.curriculum (False keeps levels fixed)
            fixed_level: start every robot on this level instead of the easy ones
            fixed_speed: always command this walking speed [m/s]
            resample_commands: on flat ground, pick a new random speed / direction every few seconds
        """
        super().__init__(num_envs, num_threads, seed, randomize, physics, push)
        cfg = HikeCfg
        self.curriculum = cfg.curriculum if curriculum is None else curriculum
        self.fixed_speed = fixed_speed
        self.resample_commands = resample_commands

        rows = cfg.flat_rows if flat_only else (rows or cfg.rows)
        self.terrain = Terrain(rows, cfg, seed=terrain_seed)
        self._init_sim(self.terrain.build_models(SimCfg), variants_per_scene=DomainRandCfg.variants_per_tile)
        n, nu = self.num_envs, self.num_actions

        # terrain assignment: each robot has a fixed row and a level that changes
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
            self.env_level = self.rng.integers(0, min(cfg.max_init_level, max_level) + 1, size=n)

        # commands
        self.commands = np.zeros((n, 3))                    # forward speed, sideways speed, yaw rate
        self.speed = np.zeros(n)
        self.move_dir = np.zeros(n)                         # flat ground: direction to walk in ...
        self.velocity_mode = np.zeros(n, dtype=bool)        # flat ground: random velocity command instead
        self.velocity_cmd = np.zeros((n, 3))
        self.progress = np.zeros(n)

        # observation
        sx, sy = np.meshgrid(np.array(cfg.scan_x), np.array(cfg.scan_y), indexing="ij")
        self._scan_x, self._scan_y = sx.ravel(), sy.ravel()
        self.num_scan = self._scan_x.size
        self.num_obs = 71 + 3 + self.num_scan
        self._noise_scale = np.concatenate(self._proprioception_noise() + [
            np.zeros(3), np.full(self.num_scan, DomainRandCfg.noise_height_scan * ObsCfg.height_scale),
        ]).astype(np.float32)
        self._init_rewards({**RewardCfg.common, **RewardCfg.hike})

    # ---------------------------------------------------------------- task
    @property
    def env_tile(self):
        return self.terrain.tile_index[self.env_row, self.env_level]

    def _ground(self, x, y):
        tile = self.env_scene.reshape((-1,) + (1,) * (np.ndim(x) - 1))
        return self.terrain.height(tile, x, y)

    def _reset_envs(self, ids):
        n, rng, cfg, terrain = ids.size, self.rng, HikeCfg, self.terrain
        self.env_scene[ids] = self.env_tile[ids]
        self._assign_models(ids)
        flat = self.is_flat[ids]
        tile = self.env_scene[ids]
        start_col = terrain.column(cfg.start_x)
        x = np.where(flat, rng.uniform(-0.5, 0.5, n), cfg.start_x + rng.uniform(-0.2, 0.2, n))
        y = np.where(flat, rng.uniform(-0.5, 0.5, n), terrain.path_y[tile, start_col] + rng.uniform(-0.3, 0.3, n))
        yaw = np.where(flat, rng.uniform(-np.pi, np.pi, n), rng.uniform(-0.4, 0.4, n))
        z = terrain.height(tile, x, y) + RobotCfg.init_height
        self._robot_reset_state(ids, x, y, z, yaw)
        self.move_dir[ids] = yaw
        self.velocity_mode[ids] = False
        self._sample_commands(ids)

    def _sample_commands(self, ids):
        cfg, rng, n = HikeCfg, self.rng, ids.size
        if self.fixed_speed is not None:
            self.speed[ids] = self.fixed_speed
        else:
            self.speed[ids] = rng.uniform(*cfg.speed, n)
            stand = self.is_flat[ids] & (rng.random(n) < cfg.stand_probability)
            self.speed[ids[stand]] = 0.0
        if self.resample_commands:
            flat = ids[self.is_flat[ids]]
            k = flat.size
            self.move_dir[flat] = rng.uniform(-np.pi, np.pi, k)
            # random velocity commands; each component is switched off now and then (pure turning, pure sideways, ...)
            self.velocity_mode[flat] = (rng.random(k) < cfg.velocity_mode) & (self.speed[flat] > 0)
            cmd = np.stack([rng.uniform(-cfg.max_back_speed, cfg.speed[1], k),
                            rng.uniform(-cfg.max_side_speed, cfg.max_side_speed, k),
                            rng.uniform(-cfg.max_yaw_rate, cfg.max_yaw_rate, k)], axis=1)
            self.velocity_cmd[flat] = cmd * (rng.random((k, 3)) > 0.25)

    def _update_task(self):
        """Navigation layer: where the trail goes -> velocity command."""
        cfg, terrain = HikeCfg, self.terrain
        due = (self.episode_length % int(cfg.resample_time_s / self.dt) == 0) & (self.episode_length > 0) & self.is_flat
        if due.any():
            self._sample_commands(np.nonzero(due)[0])

        x, y = self.base_pos[:, 0], self.base_pos[:, 1]
        tile = self.env_scene
        self.column = terrain.column(x)
        look = terrain.look_index[tile, self.column]
        to_x = terrain.path_x[look] - x
        to_y = terrain.path_y[tile, look] - y
        trail_dir = np.where(to_x > 0.05, np.arctan2(to_y, np.maximum(to_x, 1e-6)), 0.0)
        err = wrap_to_pi(np.where(self.is_flat, self.move_dir, trail_dir) - self.yaw)
        # keep walking while turning (an arc), drift sideways a little to get onto the line faster
        ahead = np.cos(err)
        self.commands[:, 0] = self.speed * np.maximum(ahead, cfg.min_turn_speed)
        self.commands[:, 1] = np.clip(self.speed * np.sin(err) * np.maximum(ahead, 0.0), -cfg.max_side_speed, cfg.max_side_speed)
        self.commands[:, 2] = np.clip(cfg.heading_gain * err, -cfg.max_yaw_rate, cfg.max_yaw_rate) * (self.speed > 0)
        self.commands[self.velocity_mode] = self.velocity_cmd[self.velocity_mode]
        self.standing = (np.abs(self.commands) < GaitCfg.stand_speed).all(axis=1)
        self.progress = np.clip((x - cfg.start_x) / (cfg.goal_x - cfg.start_x), 0.0, 1.0)

    def _check_termination(self):
        cfg, terrain = HikeCfg, self.terrain
        x, y, z = self.base_pos[:, 0], self.base_pos[:, 1], self.base_pos[:, 2]
        tile = self.env_scene
        # lowest ground under the torso and the two feet: lying on the trail = low above all three
        ground = np.minimum(terrain.height(tile, x, y),
                            terrain.height(tile[:, None], self.foot_pos[:, :, 0], self.foot_pos[:, :, 1]).min(axis=1))
        fell = (z - ground < RobotCfg.min_height) | (self.projected_gravity[:, 2] > RobotCfg.max_tilt_gravity_z)
        # off the trail: too far to the side, or down in a gap / next to the trail
        side = np.abs(y - terrain.path_y[tile, self.column]) - terrain.path_hw[tile, self.column]
        off_trail = ~self.is_flat & ((side > cfg.off_trail_margin) | (z < terrain.path_elev[tile, self.column] + 0.15))
        reached_goal = ~self.is_flat & (x >= cfg.goal_x) & ~off_trail
        time_out = (self.episode_length >= self.max_episode_length) | reached_goal
        return fell | off_trail, time_out, {"reached_goal": reached_goal, "off_trail": off_trail & ~fell}

    def _on_episode_end(self, ids, info):
        """Trail robots: made it to the end -> harder level, did not even get half way -> easier."""
        if not self.curriculum or self.terrain.num_levels == 1:
            return
        ids = ids[~self.is_flat[ids]]
        up = info["reached_goal"][ids]
        down = (self.progress[ids] < 0.5) & ~up
        level = self.env_level[ids] + up.astype(np.int64) - down.astype(np.int64)
        # robots that beat the hardest level are sent to a random level to keep practising all of them
        solved = level >= self.terrain.num_levels
        level[solved] = self.rng.integers(0, self.terrain.num_levels, int(solved.sum()))
        self.env_level[ids] = np.clip(level, 0, None)

    progress_keys = (("level", "terrain_level"), ("ok", "success"), ("off", "off_trail"))

    def curriculum_state(self):
        trail = ~self.is_flat
        return {"levels": self.env_level[trail].tolist(),
                "names": [self.terrain.row_names[r] for r in self.env_row[trail]]}

    def load_curriculum(self, state, rng):
        levels, names = np.array(state["levels"]), np.array(state["names"])
        for row, name in enumerate(self.terrain.row_names):
            ids = np.nonzero(self.env_row == row)[0]
            if ids.size and (names == name).any() and not self.terrain.row_is_flat[row]:
                self.env_level[ids] = rng.choice(levels[names == name], ids.size)

    def _task_terms(self, fell):
        cmd, sigma = self.commands, RewardCfg.tracking_sigma
        lin_err = (cmd[:, 0] - self.base_lin_vel[:, 0]) ** 2 + (cmd[:, 1] - self.base_lin_vel[:, 1]) ** 2
        return {
            "tracking_lin_vel": np.exp(-lin_err / sigma),
            "tracking_ang_vel": np.exp(-(cmd[:, 2] - self.base_ang_vel[:, 2]) ** 2 / sigma),
        }

    def _compute_obs(self):
        c, s = np.cos(self.yaw)[:, None], np.sin(self.yaw)[:, None]
        px = self.base_pos[:, 0:1] + c * self._scan_x - s * self._scan_y
        py = self.base_pos[:, 1:2] + s * self._scan_x + c * self._scan_y
        scan = self.base_pos[:, 2:3] - RobotCfg.nominal_height - self._ground(px, py)
        return self._finish_obs(self._proprioception() + [
            self.commands, np.clip(scan, -1.0, 1.0) * ObsCfg.height_scale,
        ])

    def _episode_stats(self, ids, info):
        stats = super()._episode_stats(ids, info)
        trail = ids[~self.is_flat[ids]]
        if trail.size:
            stats["success"] = float(info["reached_goal"][trail].mean())
            stats["off_trail"] = float(info["off_trail"][trail].mean())
            stats["progress"] = float(self.progress[trail].mean())
        if not self.is_flat.all():
            stats["terrain_level"] = float(self.env_level[~self.is_flat].mean())
            for row in np.unique(self.env_row[~self.is_flat]):
                stats[f"level/{row}_{self.terrain.row_names[row]}"] = float(self.env_level[self.env_row == row].mean())
        return stats


def row_by_name(name):
    """A terrain row (training or held-out) by its name."""
    for row in HikeCfg.rows + EvalCfg.unseen_rows:
        if row["name"] == name:
            return row
    raise ValueError(f"unknown terrain '{name}', choose from {terrain_names()}")


def terrain_names():
    return list(dict.fromkeys(r["name"] for r in HikeCfg.rows + EvalCfg.unseen_rows))


# Quick self-test: random actions, prints the simulation throughput
if __name__ == "__main__":
    import time

    env = HikeEnv(num_envs=1024)
    obs = env.reset()
    print(f"obs {obs.shape}, actions {env.num_actions}, threads {env.num_threads}, "
          f"tiles {env.terrain.num_tiles}, geoms {env.scene_models[0].ngeom}")
    steps = 100
    start = time.time()
    for _ in range(steps):
        obs, rew, done, info = env.step(np.random.randn(env.num_envs, env.num_actions) * 0.5)
    rate = steps * env.num_envs / (time.time() - start)
    print(f"{rate:,.0f} env steps/s  (mean reward {rew.mean():.4f}, obs finite: {np.isfinite(obs).all()})")
    env.close()
