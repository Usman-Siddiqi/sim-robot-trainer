"""
train.py

Author: Usman Siddiqi

The training loop. Each iteration:

  1. collect:  all robots act for PPOCfg.num_steps control steps
               (policy on the GPU, physics on every CPU core)
  2. learn:    one PPO update on the collected batch

Progress is printed to the console and written to TensorBoard (logs/), and
checkpoints go to models/<run name>/. Training can be stopped with Ctrl+C at
any time - the current policy is saved - and continued later with --resume.
"""

import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch

from agent import PPO, ActorCritic, get_device, load_policy, print_device_info, save_checkpoint
from config import LOG_PATH, MODEL_SAVE_PATH, PPOCfg
from environment import BipedEnv


def find_latest_model(models_dir=MODEL_SAVE_PATH):
    """Most recently written checkpoint under models_dir (searched recursively), or None."""
    newest, newest_time = None, -1.0
    for root, _, files in os.walk(models_dir):
        for f in files:
            if f.endswith(".pt"):
                path = os.path.join(root, f)
                if os.path.getmtime(path) > newest_time:
                    newest, newest_time = path, os.path.getmtime(path)
    return newest


def train(num_envs=PPOCfg.num_envs, max_iterations=PPOCfg.max_iterations, flat_only=False,
          device="auto", num_threads=None, resume=None, run_name=None, seed=0,
          save_path=MODEL_SAVE_PATH, log_path=LOG_PATH):
    """
    Train the biped.

    Args:
        num_envs: robots simulated in parallel
        max_iterations: PPO iterations to run in this session
        flat_only: flat ground only (no stairs)
        device: "auto", "cuda" or "cpu" for the neural networks
        num_threads: physics threads (default: all logical cores)
        resume: checkpoint to continue from
        run_name: name of the run directory (default: timestamp)
    Returns:
        path of the final checkpoint
    """
    torch.manual_seed(seed)
    device = get_device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        # physics and learning take turns, so both may use every core
        torch.set_num_threads(num_threads or os.cpu_count() or 1)

    task = "flat" if flat_only else "stairs"
    run_name = run_name or f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(save_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print(f"Training run '{run_name}'  (task: {task})")
    print_device_info(device)

    env = BipedEnv(num_envs=num_envs, flat_only=flat_only, num_threads=num_threads, seed=seed)
    print(f"{num_envs} robots x {PPOCfg.num_steps} steps = {num_envs * PPOCfg.num_steps:,} samples per iteration "
          f"| {env.num_threads} physics threads | obs {env.num_obs}, actions {env.num_actions}")

    start_iteration = 0
    if resume:
        policy, ckpt = load_policy(resume, device)
        if policy.num_obs != env.num_obs or policy.num_actions != env.num_actions:
            raise ValueError(f"{resume} does not fit this environment")
        ppo = PPO(policy, num_envs, device)
        if "optimizer" in ckpt:
            ppo.optimizer.load_state_dict(ckpt["optimizer"])
            ppo.lr = ckpt.get("learning_rate", ppo.lr)
        start_iteration = ckpt.get("iteration", 0)
        # carry the stairs curriculum over (resampled, so num_envs may differ between runs)
        levels = ckpt.get("terrain_levels")
        if levels and not flat_only and ckpt.get("task") == task:
            stairs = ~env.is_flat
            env.env_level[stairs] = np.random.default_rng(seed).choice(levels, int(stairs.sum()))
        print(f"Resumed from {resume} (iteration {start_iteration})")
    else:
        policy = ActorCritic(env.num_obs, env.num_actions)
        ppo = PPO(policy, num_envs, device)
    print("=" * 70)

    writer = _make_writer(os.path.join(log_path, run_name))

    def checkpoint(name, iteration):
        path = os.path.join(run_dir, name)
        levels = env.env_level[~env.is_flat].tolist()
        save_checkpoint(path, policy, ppo, iteration, {"task": task, "terrain_levels": levels})
        return path

    obs = torch.from_numpy(env.reset()).to(device)
    returns = np.zeros(num_envs)
    lengths = np.zeros(num_envs)
    return_history, length_history = deque(maxlen=200), deque(maxlen=200)
    episode_stats = {}
    total_steps = 0
    start_time = time.time()
    iteration = start_iteration
    end_iteration = start_iteration + max_iterations

    try:
        for iteration in range(start_iteration + 1, end_iteration + 1):
            # ------------------------------------------------------ collect
            t0 = time.time()
            for _ in range(PPOCfg.num_steps):
                actions = ppo.act(obs)
                obs_np, rewards, dones, info = env.step(actions.cpu().numpy())
                obs = torch.from_numpy(obs_np).to(device)
                ppo.record(
                    torch.from_numpy(rewards).to(device),
                    torch.from_numpy(dones).to(device, dtype=torch.float32),
                    torch.from_numpy(info["time_outs"]).to(device, dtype=torch.float32),
                )
                returns += rewards
                lengths += 1
                if dones.any():
                    ids = np.nonzero(dones)[0]
                    return_history.extend(returns[ids])
                    length_history.extend(lengths[ids])
                    returns[ids] = 0.0
                    lengths[ids] = 0.0
                    for k, v in info["episode"].items():
                        episode_stats.setdefault(k, deque(maxlen=20)).append(v)
            t1 = time.time()

            # -------------------------------------------------------- learn
            losses = ppo.update(obs)
            t2 = time.time()

            # ---------------------------------------------------------- log
            samples = num_envs * PPOCfg.num_steps
            total_steps += samples
            fps = samples / (t2 - t0)
            mean_return = float(np.mean(return_history)) if return_history else 0.0
            mean_length = float(np.mean(length_history)) if length_history else 0.0
            means = {k: float(np.mean(v)) for k, v in episode_stats.items()}
            if writer:
                step = iteration
                writer.add_scalar("train/mean_return", mean_return, step)
                writer.add_scalar("train/mean_episode_length", mean_length, step)
                writer.add_scalar("perf/steps_per_second", fps, step)
                writer.add_scalar("perf/collect_time", t1 - t0, step)
                writer.add_scalar("perf/learn_time", t2 - t1, step)
                for k, v in losses.items():
                    writer.add_scalar(f"loss/{k}", v, step)
                for k, v in means.items():
                    writer.add_scalar(f"episode/{k}", v, step)
            if iteration % PPOCfg.log_interval == 0 or iteration == start_iteration + 1:
                elapsed = time.time() - start_time
                done_iters = iteration - start_iteration
                eta = elapsed / done_iters * (end_iteration - iteration)
                line = (f"it {iteration:5d}/{end_iteration} | {fps:7,.0f} steps/s "
                        f"(sim {t1 - t0:4.1f}s, learn {t2 - t1:4.1f}s) | return {mean_return:7.2f} "
                        f"| ep len {mean_length:6.1f} | fell {means.get('fell', 0.0):4.2f}")
                if "terrain_level" in means:
                    line += (f" | stairs lvl {means['terrain_level']:4.2f}"
                             f" ok {means.get('stairs_success', 0.0):4.2f}")
                line += f" | std {losses['noise_std']:4.2f} | lr {losses['learning_rate']:.0e} | eta {eta / 60:5.1f} min"
                print(line, flush=True)
            if iteration % PPOCfg.save_interval == 0:
                checkpoint(f"model_{iteration}.pt", iteration)
    except KeyboardInterrupt:
        print("\nInterrupted - saving the current policy...")

    final_path = checkpoint("model_final.pt", iteration)
    if writer:
        writer.close()
    env.close()
    minutes = (time.time() - start_time) / 60
    print("=" * 70)
    print(f"Done: {total_steps:,} steps in {minutes:.1f} min. Final model: {final_path}")
    print(f"Watch it:   python main.py play --model {final_path}")
    print(f"Continue:   python main.py train --resume {final_path}")
    return final_path


def _make_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        print("TensorBoard is not installed - skipping TensorBoard logs (pip install tensorboard)")
        return None
    return SummaryWriter(log_dir)


if __name__ == "__main__":
    train()
