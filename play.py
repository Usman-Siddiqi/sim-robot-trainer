"""
play.py

Author: Usman Siddiqi

Everything you do with a trained policy:

  play()      watch the robot in the interactive MuJoCo viewer
  record()    render an episode to a video file (no window needed)
  evaluate()  measure success rates on every terrain type and stairs level
"""

import os
import time

import mujoco
import numpy as np
import torch

from agent import get_device, load_policy
from config import TerrainCfg
from environment import BipedEnv

TERRAINS = ["flat", "stairs_up", "stairs_down"]


def _single_env(policy_path, terrain, level, speed, device, seed):
    """One robot on one chosen terrain, no curriculum, no random pushes."""
    if terrain not in TERRAINS:
        raise ValueError(f"terrain must be one of {TERRAINS}")
    device = get_device(device)
    policy, ckpt = load_policy(policy_path, device)
    policy.eval()
    flat_only = terrain == "flat" and ckpt.get("task") == "flat"
    props = [float(t == terrain) for t in TerrainCfg.types]
    env = BipedEnv(num_envs=1, flat_only=flat_only, num_threads=1, seed=seed, curriculum=False,
                   fixed_level=level, terrain_proportions=None if flat_only else props,
                   push_robots=False, fixed_speed=speed, resample_commands=False)
    if policy.num_obs != env.num_obs:
        raise ValueError(f"{policy_path} was trained with a different observation size")
    return env, policy, device


def _tracking_camera(model, distance=3.0, azimuth=110.0, elevation=-12.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    cam.lookat[:] = [0.0, 0.0, 0.6]
    return cam


def play(policy_path, terrain="stairs_up", level=5, speed=0.6, device="cpu", seed=0, duration=None):
    """
    Watch a trained policy in the MuJoCo viewer. Runs in real time until the
    window is closed (or for `duration` seconds if given).

    Keys:  UP / DOWN  faster / slower      LEFT / RIGHT  turn      BACKSPACE  restart the episode
    """
    import mujoco.viewer

    env, policy, device = _single_env(policy_path, terrain, level, speed, device, seed)
    data = mujoco.MjData(env.model)
    obs = env.reset()
    env.write_data(0, data)
    state = {"reset": False}

    def on_key(key):
        if key == 265:      # up
            env.fixed_speed = min(env.fixed_speed + 0.1, 1.2)
        elif key == 264:    # down
            env.fixed_speed = max(env.fixed_speed - 0.1, 0.1)
        elif key == 263:    # left
            env.heading_target[0] += np.radians(20)
        elif key == 262:    # right
            env.heading_target[0] -= np.radians(20)
        elif key == 259:    # backspace
            state["reset"] = True
        env.commands[0, 0] = env.fixed_speed

    print(f"Playing {policy_path} on '{terrain}' (level {level}, {speed:.1f} m/s). Close the window to stop.")
    print("Keys: UP/DOWN speed, LEFT/RIGHT turn, BACKSPACE restart")
    with mujoco.viewer.launch_passive(env.model, data, key_callback=on_key) as viewer:
        cam = _tracking_camera(env.model)
        viewer.cam.type, viewer.cam.trackbodyid = cam.type, cam.trackbodyid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        started = time.time()
        while viewer.is_running() and (duration is None or time.time() - started < duration):
            tick = time.time()
            if state["reset"]:
                obs, state["reset"] = env.reset(), False
            actions = policy.act_inference(torch.from_numpy(obs).to(device))
            obs, _, done, info = env.step(actions.cpu().numpy())
            if done[0]:
                why = "fell" if info["fell"][0] else ("reached the goal" if info["reached_goal"][0] else "time is up")
                print(f"episode over: {why}")
            env.write_data(0, data)
            viewer.sync()
            time.sleep(max(0.0, env.dt - (time.time() - tick)))
    env.close()


def record(policy_path, output="walk.mp4", terrain="stairs_up", level=5, speed=0.6, seconds=12.0,
           device="cpu", seed=0, width=960, height=540, stop_when_done=True):
    """Render one episode to a video (mp4 or gif). Needs: pip install imageio imageio-ffmpeg"""
    try:
        import imageio
    except ImportError as exc:
        raise SystemExit("Recording needs imageio:  pip install imageio imageio-ffmpeg") from exc

    env, policy, device = _single_env(policy_path, terrain, level, speed, device, seed)
    data = mujoco.MjData(env.model)
    renderer = mujoco.Renderer(env.model, height, width)
    cam = _tracking_camera(env.model)
    obs = env.reset()
    frames, outcome = [], "time is up"
    every = max(1, int(round(1.0 / 25.0 / env.dt)))        # 25 fps video
    for step in range(int(seconds / env.dt)):
        actions = policy.act_inference(torch.from_numpy(obs).to(device))
        obs, _, done, info = env.step(actions.cpu().numpy())
        if done[0]:
            outcome = "fell" if info["fell"][0] else ("reached the goal" if info["reached_goal"][0] else outcome)
            if stop_when_done:
                break
        if step % every == 0:
            env.write_data(0, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
    renderer.close()
    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fps = 1.0 / (env.dt * every)
    if output.lower().endswith(".gif"):
        imageio.mimsave(output, frames, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(output, frames, fps=fps)
    print(f"Saved {len(frames)} frames to {output} ({outcome})")
    return outcome


def evaluate(policy_path, num_envs=1000, device="auto", num_threads=None, seed=123, verbose=True):
    """
    Run one full episode per robot with the deterministic policy and report, for
    every terrain type and stairs level, how often the robot fell / succeeded.

    Returns a dict: results[terrain][level] = {"success", "fell", "speed_error", "n"}
    (flat ground uses level 0 and counts surviving the whole episode as success).
    """
    device = get_device(device)
    policy, ckpt = load_policy(policy_path, device)
    policy.eval()
    flat_only = ckpt.get("task") == "flat"
    env = BipedEnv(num_envs=num_envs, flat_only=flat_only, num_threads=num_threads, seed=seed,
                   curriculum=False, push_robots=False,
                   terrain_proportions=None if flat_only else [0.1, 0.45, 0.45])
    stairs = np.nonzero(~env.is_flat)[0]
    env.env_level[stairs] = np.arange(stairs.size) % env.terrain.num_levels
    env.env_level[env.is_flat] = 0
    obs = env.reset()

    running = np.ones(num_envs, dtype=bool)         # still in its first episode
    success = np.zeros(num_envs, dtype=bool)
    fell = np.zeros(num_envs, dtype=bool)
    speed_err = np.zeros(num_envs)
    steps = np.zeros(num_envs)
    levels = env.env_level.copy()
    for _ in range(env.max_episode_length + 1):
        cmd_speed = env.commands[:, 0].copy()
        actions = policy.act_inference(torch.from_numpy(obs).to(device))
        obs, _, done, info = env.step(actions.cpu().numpy())
        ended = running & done
        fell |= ended & info["fell"]
        success |= ended & ~info["fell"] & (info["reached_goal"] | env.is_flat)
        active = running & ~done
        speed_err[active] += np.abs(env.base_lin_vel[active, 0] - cmd_speed[active])
        steps[active] += 1
        running &= ~done
        if not running.any():
            break
    env.close()

    results = {}
    for row, name in enumerate(["flat"] if flat_only else TerrainCfg.types):
        results[name] = {}
        for level in range(env.terrain.num_levels if name != "flat" else 1):
            mask = (env.env_row == row) & (levels == level)
            if mask.any():
                results[name][level] = {
                    "success": float(success[mask].mean()), "fell": float(fell[mask].mean()),
                    "speed_error": float((speed_err[mask] / np.maximum(steps[mask], 1)).mean()),
                    "n": int(mask.sum()),
                }
    if verbose:
        print(f"\nEvaluation of {policy_path}  ({num_envs} robots, one episode each, no exploration noise)")
        print(f"{'terrain':<12} {'level':>5} {'step [cm]':>9} {'success':>8} {'fell':>6} {'speed err [m/s]':>16} {'robots':>7}")
        for name, per_level in results.items():
            for level, r in per_level.items():
                step_cm = "-" if name == "flat" else f"{env.terrain.step_heights[level] * 100:.1f}"
                print(f"{name:<12} {level:>5} {step_cm:>9} {r['success']:>8.0%} {r['fell']:>6.0%} "
                      f"{r['speed_error']:>16.2f} {r['n']:>7}")
    return results
