"""
main.py

Author: Usman Siddiqi

Command line entry point of sim-robot-trainer. There are two tasks:

    hike    follow winding mountain trails: rock steps, stepping stones, ridges, boulders
    sort    pick up coloured boxes of different weights and carry each to the table of its colour

    python main.py train --task hike        train (add --resume to continue from the newest model)
    python main.py train --task sort
    python main.py play  --task hike        watch a policy in the viewer (the bundled one if you have none)
    python main.py play  --task sort
    python main.py eval  --task hike        success rates, including conditions it never trained in
    python main.py record --task sort --output sort.mp4
    python main.py bench --task hike        how fast is the simulation on this machine?
    python main.py list                     list saved models

Run `python main.py <command> --help` for the options of each command.
"""

import argparse
import os
import sys
import time

from config import MODEL_SAVE_PATH, PPOCfg

TASKS = ("hike", "sort")
_HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAINED = {task: os.path.join(_HERE, "pretrained", f"{task}.pt") for task in TASKS}


def model_task(path):
    """Task a checkpoint was trained for (None if it cannot be read)."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True).get("task", "hike")
    except Exception:
        return None


def list_models(models_dir=MODEL_SAVE_PATH):
    """Print all saved checkpoints, newest first. Returns their paths."""
    found = []
    for root, _, files in os.walk(models_dir):
        found += [os.path.join(root, f) for f in files if f.endswith(".pt")]
    found.sort(key=os.path.getmtime, reverse=True)
    if not found:
        print(f"No trained models in '{models_dir}' yet. Train one with: python main.py train --task hike")
        return found
    print(f"Models in '{models_dir}' (newest first):")
    for i, path in enumerate(found, 1):
        size = os.path.getsize(path) / 2**20
        print(f"  {i:3d}. {path}  ({model_task(path)}, {size:.1f} MB, {time.ctime(os.path.getmtime(path))})")
    return found


def resolve_model(path, task):
    """Model given on the command line, else the newest trained one for the task, else the bundled one."""
    if path:
        if not os.path.exists(path):
            sys.exit(f"Model not found: {path}")
        return path
    from train import find_latest_model
    latest = find_latest_model(task=task)
    if latest:
        print(f"Using the newest {task} model: {latest}")
        return latest
    if os.path.exists(PRETRAINED[task]):
        print(f"No trained {task} model found - using the bundled one: {PRETRAINED[task]}")
        return PRETRAINED[task]
    sys.exit(f"No {task} model found. Train one first: python main.py train --task {task}")


def bench(task, num_envs, num_threads):
    """Measure raw simulation throughput (random actions) and report the PyTorch device."""
    import numpy as np
    from agent import get_device, print_device_info
    from train import make_env

    print_device_info(get_device("auto"))
    env = make_env(task, num_envs=num_envs, num_threads=num_threads)
    env.reset()
    actions = np.random.default_rng(0).normal(0.0, 0.3, (50, num_envs, env.num_actions))
    for a in actions[:5]:
        env.step(a)
    start = time.time()
    for a in actions:
        env.step(a)
    rate = len(actions) * num_envs / (time.time() - start)
    env.close()
    print(f"{task}: {num_envs} robots on {env.num_threads} threads: {rate:,.0f} simulation steps/s "
          f"({rate * env.dt:,.0f}x real time)")
    per_iter = num_envs * PPOCfg.num_steps / rate
    print(f"=> roughly {per_iter:.1f} s of simulation per training iteration, plus the PPO update")


def main():
    parser = argparse.ArgumentParser(
        description="Sim Robot Trainer - teach a simulated humanoid to hike and to sort boxes with PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_task(p):
        p.add_argument("--task", default="hike", choices=TASKS, help="hike (default) or sort")

    p = sub.add_parser("train", help="train a policy")
    add_task(p)
    p.add_argument("--flat", action="store_true", help="hike: flat ground only, no trails (quick sanity check)")
    p.add_argument("--num_envs", type=int, default=PPOCfg.num_envs, help="robots simulated in parallel")
    p.add_argument("--iterations", type=int, default=PPOCfg.max_iterations, help="PPO iterations to run")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu (neural networks)")
    p.add_argument("--threads", type=int, default=None, help="physics threads (default: all cores)")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="continue from a checkpoint (no path = the newest model of the task, else the bundled one)")
    p.add_argument("--name", default=None, help="run name (default: timestamp)")
    p.add_argument("--seed", type=int, default=0)

    def add_scene_args(p):
        add_task(p)
        p.add_argument("--model", default=None, help="checkpoint (default: newest of the task, else the bundled one)")
        p.add_argument("--terrain", default="trail", metavar="NAME",
                       help="hike: a training trail (flat, rocky, climb, stones, ridge, trail) or a held-out one "
                            "(logs, terraces, rubble, staircase, switchbacks, new_trail)")
        p.add_argument("--level", type=int, default=6, help="hike: difficulty level 0-9")
        p.add_argument("--speed", type=float, default=0.6, help="hike: commanded walking speed [m/s]")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("play", help="watch a trained policy in the MuJoCo viewer")
    add_scene_args(p)

    p = sub.add_parser("record", help="render an episode to a video file")
    add_scene_args(p)
    p.add_argument("--output", default="robot.mp4", help="output file (.mp4 or .gif)")
    p.add_argument("--seconds", type=float, default=None, help="default: one whole episode")

    p = sub.add_parser("eval", help="measure success rates of a trained policy")
    add_task(p)
    p.add_argument("--model", default=None)
    p.add_argument("--suite", default="all", choices=["all", "trained", "unseen"],
                   help="trained = training conditions, unseen = held-out trails / boxes / physics")
    p.add_argument("--num_envs", type=int, default=None, help="robots per test")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)

    p = sub.add_parser("bench", help="measure simulation speed on this machine")
    add_task(p)
    p.add_argument("--num_envs", type=int, default=PPOCfg.num_envs)
    p.add_argument("--threads", type=int, default=None)

    p = sub.add_parser("list", help="list saved models")
    p.add_argument("--models_dir", default=MODEL_SAVE_PATH)

    args = parser.parse_args()

    if args.command == "train":
        from train import train
        resume = args.resume
        if resume == "latest":
            resume = resolve_model(None, args.task)
        elif resume and not os.path.exists(resume):
            sys.exit(f"Model not found: {resume}")
        train(task=args.task, num_envs=args.num_envs, max_iterations=args.iterations, flat_only=args.flat,
              device=args.device, num_threads=args.threads, resume=resume, run_name=args.name, seed=args.seed)
    elif args.command in ("play", "record", "eval"):
        model = resolve_model(args.model, args.task)
        if args.command == "play":
            from play import play
            play(model, args.terrain, args.level, args.speed, seed=args.seed)
        elif args.command == "record":
            from play import record
            record(model, args.output, args.terrain, args.level, args.speed, seconds=args.seconds, seed=args.seed)
        else:
            from play import evaluate
            evaluate(model, args.num_envs, args.device, args.threads, suite=args.suite)
    elif args.command == "bench":
        bench(args.task, args.num_envs, args.threads)
    elif args.command == "list":
        list_models(args.models_dir)


if __name__ == "__main__":
    main()
