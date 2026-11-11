"""
play.py

Author: Usman Siddiqi

Everything you do with a trained policy:

  play()      watch the robot in the interactive MuJoCo viewer
  record()    render an episode to a video file (no window needed)
  evaluate()  measure success rates, including conditions the policy never trained in

The task (hike or sort) is read from the checkpoint.
"""

import os
import threading
import time

import mujoco
import numpy as np
import torch

from agent import get_device, load_policy
from config import EvalCfg, HikeCfg, SortCfg


# ----------------------------------------------------------------- scenes
def load(policy_path, device="cpu"):
    """-> (policy, task name, device)"""
    device = get_device(device)
    policy, ckpt = load_policy(policy_path, device)
    policy.eval()
    return policy, ckpt.get("task", "hike"), device


def make_scene(task, terrain="trail", level=5, speed=0.6, seed=0, physics=None, push=None, box_mass=None):
    """One nominal robot for watching: no curriculum, no sensor noise, no pushes unless asked for."""
    if task == "sort":
        from env_sort import SortEnv
        return SortEnv(num_envs=1, num_threads=1, seed=seed, randomize=False, physics=physics, push=push,
                       box_mass=box_mass)
    from env_hike import HikeEnv, row_by_name
    flat = dict(HikeCfg.flat_rows[0], weight=0.0)
    rows = None if terrain == "flat" else [flat, dict(row_by_name(terrain), weight=1.0)]
    return HikeEnv(num_envs=1, flat_only=terrain == "flat", num_threads=1, seed=seed, rows=rows, terrain_seed=seed,
                   curriculum=False, fixed_level=level, randomize=False, physics=physics, push=push,
                   fixed_speed=speed, resample_commands=terrain != "flat")


def make_camera(env, model):
    cam = mujoco.MjvCamera()
    if env.task == "sort":                                  # fixed view of the whole room
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.1, 0.0, 0.65]
        cam.distance, cam.azimuth, cam.elevation = 4.6, 150.0, -28.0
    else:                                                   # follow the hiker
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        cam.distance, cam.azimuth, cam.elevation = 4.2, 125.0, -26.0
        cam.lookat[:] = [0.0, 0.0, 0.6]
    return cam


class Scene:
    """A policy acting in a one-robot environment, with the bookkeeping the viewer and the recorder share."""

    def __init__(self, policy, env, device):
        self.policy, self.env, self.device = policy, env, device
        if policy.num_obs != env.num_obs:
            raise ValueError(f"this policy was trained with a different observation size ({policy.num_obs}) "
                             f"than the '{env.task}' task uses ({env.num_obs})")
        self.obs = env.reset()
        self.model = env.model_of(0)
        self.data = mujoco.MjData(self.model)
        self.sorted_boxes = []                              # sort: boxes already put down (pos, mat, size, rgb)
        self.events = []
        self._remember_box()
        self.sync()

    def _remember_box(self):
        env = self.env
        if env.task != "sort":
            return
        qpos = env.state[0, env._box_qpos]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, qpos[3:7])
        self._box = (qpos[0:3].copy(), mat, env.box_half[0].copy(), env.colors[env.box_color[0]].copy())
        self._count = int(env.sorted_count[0])

    def step(self):
        env = self.env
        actions = self.policy.act_inference(torch.from_numpy(self.obs).to(self.device))
        self.obs, _, done, info = env.step(actions.cpu().numpy())
        event = None
        if env.task == "sort":
            if info["sorted"][0]:
                self.sorted_boxes.append(self._box)
                event = f"box {len(self.sorted_boxes)} sorted"
            elif info["dropped"][0]:
                event = "dropped the box"
            elif info["wrong_table"][0]:
                event = "wrong table"
        if done[0]:
            if info["fell"][0]:
                event = "left the trail" if info.get("off_trail", [False])[0] else "fell"
            elif info.get("reached_goal", [False])[0]:
                event = "reached the end of the trail"
            else:
                event = "time is up"
            self.sorted_boxes = []
        self._remember_box()
        if event:
            self.events.append(event)
        return done[0], event

    def sync(self):
        """Copy the robot's state into self.data (and self.model, which changes with the trail / box)."""
        self.model = self.env.model_of(0)
        self.env.write_data(0, self.data)

    def draw_extras(self, scn):
        """Add the boxes that were already sorted to a scene (they are not simulated any more)."""
        rim, thick = SortCfg.rim_width, SortCfg.rim_thickness
        for pos, mat, size, rgb in self.sorted_boxes:
            up = mat.reshape(3, 3)[:, 2]
            for centre, half, colour in [(pos, size, rgb),
                                         (pos + up * (size[2] - thick / 2), size + [rim, rim, thick / 2 - size[2]], 0.8 * rgb)]:
                if scn.ngeom >= scn.maxgeom:
                    return
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, dtype=np.float64),
                                    np.asarray(centre, dtype=np.float64), mat, np.array([*colour, 1.0], dtype=np.float32))
                scn.ngeom += 1


# ------------------------------------------------------------------ viewer
def play(policy_path, terrain="trail", level=5, speed=0.6, device="cpu", seed=0, duration=None):
    """
    Watch a trained policy in the MuJoCo viewer. Runs in real time until the
    window is closed (or for `duration` seconds if given).

    Keys:  BACKSPACE  restart the episode      (hike) UP / DOWN  faster / slower
    """
    import mujoco.viewer

    policy, task, device = load(policy_path, device)
    env = make_scene(task, terrain, level, speed, seed)
    scene = Scene(policy, env, device)
    state = {"reset": False}

    def on_key(key):
        if key == 259:      # backspace
            state["reset"] = True
        elif task == "hike" and key in (265, 264):          # up / down
            env.fixed_speed = float(np.clip(env.fixed_speed + (0.1 if key == 265 else -0.1), 0.0, 1.0))
            env.speed[0] = env.fixed_speed

    what = f"'{terrain}' (level {level}, {speed:.1f} m/s)" if task == "hike" else "the sorting room"
    print(f"Playing {policy_path} in {what}. Close the window to stop.")
    print("Keys: BACKSPACE restart" + (", UP/DOWN speed" if task == "hike" else ""))
    threads_before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(scene.model, scene.data, key_callback=on_key)
    viewer_threads = [t for t in threading.enumerate() if t not in threads_before]
    try:
        cam = make_camera(env, scene.model)
        viewer.cam.type, viewer.cam.trackbodyid = cam.type, cam.trackbodyid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        viewer.cam.lookat[:] = cam.lookat
        started = time.time()
        while viewer.is_running() and (duration is None or time.time() - started < duration):
            tick = time.time()
            if state["reset"]:
                scene.obs, state["reset"], scene.sorted_boxes = env.reset(), False, []
            _, event = scene.step()
            if event:
                print(event)
            with viewer.lock():
                scene.sync()
                viewer.user_scn.ngeom = 0
                scene.draw_extras(viewer.user_scn)
            viewer.sync()
            time.sleep(max(0.0, env.dt - (time.time() - tick)))
    finally:
        viewer.close()
        # the viewer shuts down on its own thread; wait for it so Python does not exit underneath it
        for thread in viewer_threads:
            thread.join(timeout=10)
        env.close()


# ---------------------------------------------------------------- recorder
def render_episode(scene, seconds, width=960, height=540, fps=25, stop_when_done=True, on_frame=None):
    """Run a Scene and render it. Returns (frames, list of events)."""
    env = scene.env
    renderer = mujoco.Renderer(scene.model, height, width)
    cam = make_camera(env, scene.model)
    every = max(1, int(round(1.0 / fps / env.dt)))
    frames = []
    for step in range(int(seconds / env.dt)):
        done, event = scene.step()
        if done and stop_when_done:
            break
        if step % every == 0:
            scene.sync()
            renderer.update_scene(scene.data, camera=cam)
            scene.draw_extras(renderer.scene)
            frame = renderer.render().copy()
            frames.append(on_frame(frame, scene, event) if on_frame else frame)
    renderer.close()
    return frames, scene.events


def record(policy_path, output="robot.mp4", terrain="trail", level=5, speed=0.6, seconds=None,
           device="cpu", seed=0, width=960, height=540):
    """Render one episode to a video (mp4 or gif). Needs: pip install imageio imageio-ffmpeg"""
    try:
        import imageio
    except ImportError as exc:
        raise SystemExit("Recording needs imageio:  pip install imageio imageio-ffmpeg") from exc

    policy, task, device = load(policy_path, device)
    env = make_scene(task, terrain, level, speed, seed)
    scene = Scene(policy, env, device)
    frames, events = render_episode(scene, seconds or env.episode_length_s, width, height)
    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    if output.lower().endswith(".gif"):
        imageio.mimsave(output, frames, duration=1.0 / 25, loop=0)
    else:
        imageio.mimsave(output, frames, fps=25)
    print(f"Saved {len(frames)} frames to {output} ({', '.join(events) or 'time is up'})")
    return events


# -------------------------------------------------------------- evaluation
def _act(policy, obs, device):
    return policy.act_inference(torch.from_numpy(obs).to(device)).cpu().numpy()


def _hike_episodes(policy, env, device):
    """One episode per robot with the deterministic policy -> (success, fell) per robot."""
    for row in np.unique(env.env_row[~env.is_flat]):            # spread each row evenly over the levels
        ids = np.nonzero(env.env_row == row)[0]
        env.env_level[ids] = np.arange(ids.size) % env.terrain.num_levels
    env.env_level[env.is_flat] = 0
    levels = env.env_level.copy()
    obs = env.reset()
    n = env.num_envs
    running = np.ones(n, dtype=bool)                            # still in its first episode
    success, fell = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for _ in range(env.max_episode_length + 1):
        obs, _, done, info = env.step(_act(policy, obs, device))
        ended = running & done
        fell |= ended & info["fell"]
        success |= ended & ~info["fell"] & (info["reached_goal"] | env.is_flat)
        running &= ~done
        if not running.any():
            break
    return levels, success, fell


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
    print(f"  {'trail':<14} {'success':>7} {'failed':>6}   success per level: {header}")
    for name, r in table.items():
        cells = " ".join(f"{v * 100:3.0f} " for v in r["levels"]) if len(r["levels"]) > 1 else "(no levels)"
        print(f"  {name:<14} {r['success']:>7.0%} {r['fell']:>6.0%}                      {cells}")


def evaluate_hike(policy, device, num_envs=2000, num_threads=None, seed=123, suite="all", verbose=True):
    from env_hike import HikeEnv

    common = dict(num_envs=num_envs, num_threads=num_threads, seed=seed, curriculum=False)
    results = {}
    if suite in ("trained", "all"):
        env = HikeEnv(randomize=False, **common)
        results["trained"] = _per_terrain(env, *_hike_episodes(policy, env, device))
        if verbose:
            _print_terrain_table("TRAINING TRAILS (nominal robot)", results["trained"], env.terrain.num_levels)
        env.close()

    if suite in ("unseen", "all"):
        env = HikeEnv(randomize=False, rows=EvalCfg.unseen_rows, terrain_seed=EvalCfg.unseen_seed, **common)
        results["unseen_trails"] = _per_terrain(env, *_hike_episodes(policy, env, device))
        if verbose:
            _print_terrain_table("UNSEEN TRAILS (sections and layouts never used in training)",
                                 results["unseen_trails"], env.terrain.num_levels)
        env.close()

        results["unseen_conditions"] = {}
        names = list(dict.fromkeys(row["name"] for row in HikeCfg.rows))
        if verbose:
            print("\nUNSEEN CONDITIONS (training trails, physics outside the training ranges)")
            print(f"  success, all levels          {'condition':<14}" + "".join(f"{n:>8}" for n in names)
                  + f"{'overall':>9} {'failed':>6}")
        for cond in EvalCfg.unseen_conditions:
            env = HikeEnv(randomize=False, physics=cond.get("physics"), push=cond.get("push"), **common)
            levels, success, fell = _hike_episodes(policy, env, device)
            r = {"overall": float(success.mean()), "fell": float(fell.mean()),
                 "per_terrain": _per_terrain(env, levels, success, fell)}
            results["unseen_conditions"][cond["name"]] = r
            if verbose:
                cells = "".join(f"{r['per_terrain'][n]['success']:>8.0%}" for n in names)
                print(f"  {cond['name']:<43}{cells}{r['overall']:>9.0%} {r['fell']:>6.0%}")
            env.close()
        if verbose:
            avg = np.mean([r["success"] for name, r in results["unseen_trails"].items() if name != "flat"])
            cond = np.mean([r["overall"] for r in results["unseen_conditions"].values()])
            print(f"\nSummary: {avg:.0%} average success on unseen trails, {cond:.0%} under unseen conditions")
    return results


def _sort_episodes(policy, env, device):
    """One 30 s episode per robot -> dict of per-robot arrays."""
    obs = env.reset()
    n = env.num_envs
    running = np.ones(n, dtype=bool)
    out = {"sorted": np.zeros(n), "dropped": np.zeros(n), "wrong": np.zeros(n), "fell": np.zeros(n, dtype=bool),
           "first_mass": env.box_masses[env.box_mass_index].copy(), "first_sorted": np.zeros(n, dtype=bool),
           "first_done": np.zeros(n, dtype=bool), "first_time": np.full(n, np.nan)}
    for step in range(env.max_episode_length + 1):
        obs, _, done, info = env.step(_act(policy, obs, device))
        first = running & info["sorted"] & ~out["first_done"]
        out["first_time"][first] = (step + 1) * env.dt
        out["first_sorted"] |= first
        out["first_done"] |= running & (info["sorted"] | info["dropped"] | info["wrong_table"])
        out["sorted"] += running & info["sorted"]
        out["dropped"] += running & info["dropped"]
        out["wrong"] += running & info["wrong_table"]
        out["fell"] |= running & done & info["fell"]
        running &= ~done
        if not running.any():
            break
    return out


def _sort_summary(r):
    return {"boxes": float(r["sorted"].mean()), "any": float((r["sorted"] > 0).mean()), "fell": float(r["fell"].mean()),
            "dropped": float(r["dropped"].mean()), "wrong": float(r["wrong"].mean())}


def evaluate_sort(policy, device, num_envs=1000, num_threads=None, seed=123, suite="all", verbose=True):
    from env_sort import SortEnv

    common = dict(num_envs=num_envs, num_threads=num_threads, seed=seed)
    results = {}
    row = lambda name, s: (f"  {name:<44}{s['boxes']:>7.2f}{s['any']:>12.0%}{s['fell']:>8.0%}"
                           f"{s['dropped']:>9.2f}{s['wrong']:>8.2f}")
    header = (f"  {'':<44}{'boxes':>7}{'>= 1 box':>12}{'robot':>8}{'boxes':>9}{'wrong':>8}\n"
              f"  {'':<44}{'sorted':>7}{'sorted':>12}{'fell':>8}{'dropped':>9}{'table':>8}      (per 30 s episode)")
    if suite in ("trained", "all"):
        env = SortEnv(randomize=False, **common)
        r = _sort_episodes(policy, env, device)
        results["trained"] = _sort_summary(r)
        results["by_mass"] = {float(m): float(r["first_sorted"][r["first_mass"] == m].mean()) for m in env.box_masses}
        if verbose:
            print(f"\nTRAINING CONDITIONS (nominal robot, boxes {env.box_masses[0]:.1f} - {env.box_masses[-1]:.1f} kg)")
            print(header)
            print(row("sorting room", results["trained"]))
            print("  first box sorted, by weight:  " + "   ".join(f"{m:.1f} kg {v:.0%}" for m, v in results["by_mass"].items()))
            print(f"  time to sort the first box: {np.nanmean(r['first_time']):.1f} s on average")
        env.close()
    if suite in ("unseen", "all"):
        results["unseen_conditions"] = {}
        if verbose:
            print("\nUNSEEN CONDITIONS (outside the training ranges)")
            print(header)
        for cond in EvalCfg.sort_unseen_conditions:
            env = SortEnv(randomize=False, physics=cond.get("physics"), push=cond.get("push"),
                          box_mass=cond.get("box_mass"), **common)
            s = _sort_summary(_sort_episodes(policy, env, device))
            results["unseen_conditions"][cond["name"]] = s
            if verbose:
                print(row(cond["name"], s))
            env.close()
    return results


def evaluate(policy_path, num_envs=None, device="auto", num_threads=None, seed=123, suite="all", verbose=True):
    """
    Test a policy with exploration noise switched off, one episode per robot.

    suite "trained":  the training conditions with the nominal robot
    suite "unseen":   held-out conditions the policy never trained on (EvalCfg)
    suite "all":      both

    hike: success = reaching the end of the trail (on flat ground: not falling).
    sort: boxes sorted per 30 s episode and how episodes ended.
    Returns a dict with all numbers.
    """
    policy, task, device = load(policy_path, device)
    if verbose:
        print(f"Evaluating {policy_path}  (task: {task}, one episode per robot, no exploration noise)")
    if task == "sort":
        return evaluate_sort(policy, device, num_envs or 1000, num_threads, seed, suite, verbose)
    return evaluate_hike(policy, device, num_envs or 2000, num_threads, seed, suite, verbose)
