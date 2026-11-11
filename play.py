"""
play.py

Author: Usman Siddiqi

Everything you do with a trained policy:

  play()      watch the robot in the interactive MuJoCo viewer
  record()    render an episode to a video file (no window needed)
  evaluate()  measure success rates on every terrain type and stairs level
"""

import os
import threading
import time

import mujoco
import numpy as np
import torch

from agent import get_device, load_policy
from config import EvalCfg, TerrainCfg
from environment import BipedEnv


def terrain_names():
    """All terrain names that can be shown: the training terrains first, then the held-out ones."""
    names = []
    for row in TerrainCfg.rows + EvalCfg.unseen_terrains:
        if row["name"] not in names:
            names.append(row["name"])
    return names


def _single_env(policy_path, terrain, level, speed, device, seed):
    """One nominal robot on one chosen terrain: no curriculum, no pushes, no sensor noise."""
    rows = [r for r in TerrainCfg.rows + EvalCfg.unseen_terrains if r["name"] == terrain]
    if not rows:
        raise ValueError(f"unknown terrain '{terrain}', choose from {terrain_names()}")
    device = get_device(device)
    policy, _ = load_policy(policy_path, device)
    policy.eval()
    flat = dict(TerrainCfg.flat_rows[0], weight=0.0)
    env = BipedEnv(num_envs=1, flat_only=terrain == "flat", num_threads=1, seed=seed,
                   terrain_rows=[flat, dict(rows[0], weight=1.0)], terrain_seed=seed,
                   curriculum=False, fixed_level=level, randomize=False,
                   fixed_speed=speed, resample_commands=False)
    if policy.num_obs != env.num_obs:
        raise ValueError(f"{policy_path} was trained with a different observation size")
    return env, policy, device


def _tracking_camera(model, distance=3.4, azimuth=115.0, elevation=-24.0):
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
    threads_before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(env.model, data, key_callback=on_key)
    viewer_threads = [t for t in threading.enumerate() if t not in threads_before]
    try:
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
    finally:
        viewer.close()
        # the viewer shuts down on its own thread; wait for it so Python does not exit underneath it
        for thread in viewer_threads:
            thread.join(timeout=10)
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


def _run_episodes(policy, env, device):
    """One episode per robot with the deterministic policy -> (success, fell, speed error) per robot."""
    for row in np.unique(env.env_row[~env.is_flat]):            # spread each row evenly over the levels
        ids = np.nonzero(env.env_row == row)[0]
        env.env_level[ids] = np.arange(ids.size) % env.terrain.num_levels
    env.env_level[env.is_flat] = 0
    obs = env.reset()
    n = env.num_envs
    running = np.ones(n, dtype=bool)                            # still in its first episode
    success, fell = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    speed_err, steps = np.zeros(n), np.zeros(n)
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
    return success, fell, speed_err / np.maximum(steps, 1)


def _per_terrain(env, levels, success, fell):
    """Aggregate per terrain name -> {"levels": [success per level], "success", "fell", "n"}."""
    out = {}
    for name in dict.fromkeys(env.terrain.row_names):
        rows = [r for r, row_name in enumerate(env.terrain.row_names) if row_name == name]
        mask = np.isin(env.env_row, rows)
        if not mask.any():
            continue
        if env.terrain.row_is_flat[rows[0]]:
            per_level = [float(success[mask].mean())]
        else:
            per_level = [float(success[mask & (levels == lv)].mean()) for lv in range(env.terrain.num_levels)]
        out[name] = {"levels": per_level, "success": float(success[mask].mean()),
                     "fell": float(fell[mask].mean()), "n": int(mask.sum())}
    return out


def _print_terrain_table(title, table, num_levels):
    print(f"\n{title}")
    header = " ".join(f"L{lv:<3}" for lv in range(num_levels))
    print(f"  {'terrain':<24} {'success':>7} {'fell':>5}   success per level: {header}")
    for name, r in table.items():
        cells = " ".join(f"{v * 100:3.0f} " for v in r["levels"]) if len(r["levels"]) > 1 else "(no levels)"
        print(f"  {name:<24} {r['success']:>7.0%} {r['fell']:>5.0%}                      {cells}")


def evaluate(policy_path, num_envs=2000, device="auto", num_threads=None, seed=123, suite="all", verbose=True):
    """
    Test a policy with exploration noise switched off, one episode per robot.

    suite "trained":  the training terrains with the nominal robot
    suite "unseen":   held-out conditions the policy never trained on (EvalCfg):
                      new terrain types, and the training terrains with physics
                      or commands outside the training ranges
    suite "all":      both

    Success = crossing the whole obstacle tile (or, on flat ground, not falling
    for a whole episode). Returns a dict with all numbers.
    """
    device = get_device(device)
    policy, _ = load_policy(policy_path, device)
    policy.eval()
    common = dict(num_envs=num_envs, num_threads=num_threads, seed=seed, curriculum=False)
    results = {}
    if verbose:
        print(f"Evaluating {policy_path}  ({num_envs} robots per test, one episode each, no exploration noise)")

    if suite in ("trained", "all"):
        env = BipedEnv(randomize=False, **common)
        success, fell, _ = _run_episodes(policy, env, device)
        results["trained"] = _per_terrain(env, env.env_level.copy(), success, fell)
        if verbose:
            _print_terrain_table("TRAINING TERRAINS (nominal robot)", results["trained"], env.terrain.num_levels)
        env.close()

    if suite in ("unseen", "all"):
        env = BipedEnv(randomize=False, terrain_rows=EvalCfg.unseen_terrains, terrain_seed=seed, **common)
        success, fell, _ = _run_episodes(policy, env, device)
        results["unseen_terrains"] = _per_terrain(env, env.env_level.copy(), success, fell)
        if verbose:
            _print_terrain_table("UNSEEN TERRAINS (never used in training)", results["unseen_terrains"],
                                 env.terrain.num_levels)
        env.close()

        results["unseen_conditions"] = {}
        names = list(dict.fromkeys(row["name"] for row in TerrainCfg.rows))
        if verbose:
            print("\nUNSEEN CONDITIONS (training terrains, physics / commands outside the training ranges)")
            print(f"  success, all levels          {'condition':<14}" + "".join(f"{n:>12}" for n in names)
                  + f"{'overall':>9} {'fell':>5}")
        for cond in EvalCfg.unseen_conditions:
            env = BipedEnv(randomize=False, physics=cond.get("physics"), push=cond.get("push"),
                           heading_offset=cond.get("heading_offset"), **common)
            success, fell, _ = _run_episodes(policy, env, device)
            r = {"overall": float(success.mean()), "fell": float(fell.mean()),
                 "per_terrain": _per_terrain(env, env.env_level.copy(), success, fell)}
            results["unseen_conditions"][cond["name"]] = r
            if verbose:
                cells = "".join(f"{r['per_terrain'][n]['success']:>12.0%}" for n in names)
                print(f"  {cond['name']:<43}{cells}{r['overall']:>9.0%} {r['fell']:>5.0%}")
            env.close()

    if verbose and "unseen_terrains" in results:
        avg = np.mean([r["success"] for r in results["unseen_terrains"].values()])
        cond = np.mean([r["overall"] for r in results["unseen_conditions"].values()])
        print(f"\nSummary: {avg:.0%} average success on unseen terrains, {cond:.0%} under unseen conditions")
    return results
