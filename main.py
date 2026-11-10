"""
main.py

Author: Usman Siddiqi

Command line entry point of sim-robot-trainer.

    python main.py train                 train on flat ground + stairs (the full task)
    python main.py train --flat          train on flat ground only (quick sanity check)
    python main.py play                  watch the newest trained model in the viewer
    python main.py eval                  success rate on every terrain / stairs level
    python main.py record --output walk.mp4
    python main.py bench                 how fast is the simulation on this machine?
    python main.py list                  list saved models

Run `python main.py <command> --help` for the options of each command.
"""

import argparse
import os
import sys
import time

from config import MODEL_SAVE_PATH, PPOCfg

PRETRAINED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", "biped_stairs.pt")


def list_models(models_dir=MODEL_SAVE_PATH):
    """Print all saved checkpoints, newest first. Returns their paths."""
    found = []
    for root, _, files in os.walk(models_dir):
        found += [os.path.join(root, f) for f in files if f.endswith(".pt")]
    found.sort(key=os.path.getmtime, reverse=True)
    if not found:
        print(f"No trained models in '{models_dir}' yet. Train one with: python main.py train")
        return found
    print(f"Models in '{models_dir}' (newest first):")
    for i, path in enumerate(found, 1):
        size = os.path.getsize(path) / 2**20
        print(f"  {i:3d}. {path}  ({size:.1f} MB, {time.ctime(os.path.getmtime(path))})")
    return found


def resolve_model(path):
    """Model given on the command line, else the newest trained one, else the bundled pretrained one."""
    if path:
        if not os.path.exists(path):
            sys.exit(f"Model not found: {path}")
        return path
    from train import find_latest_model
    latest = find_latest_model()
    if latest:
        print(f"Using the newest model: {latest}")
        return latest
    if os.path.exists(PRETRAINED):
        print(f"No trained model found - using the bundled pretrained policy: {PRETRAINED}")
        return PRETRAINED
    sys.exit("No trained model found. Train one first: python main.py train")


def bench(num_envs, num_threads):
    """Measure raw simulation throughput (random actions) and report the PyTorch device."""
    import numpy as np
    from agent import get_device, print_device_info
    from environment import BipedEnv

    print_device_info(get_device("auto"))
    env = BipedEnv(num_envs=num_envs, num_threads=num_threads)
    env.reset()
    actions = np.random.default_rng(0).normal(0.0, 0.3, (50, num_envs, env.num_actions))
    for a in actions[:5]:
        env.step(a)
    start = time.time()
    for a in actions:
        env.step(a)
    rate = len(actions) * num_envs / (time.time() - start)
    env.close()
    print(f"{num_envs} robots on {env.num_threads} threads: {rate:,.0f} simulation steps/s "
          f"({rate * env.dt:,.0f}x real time)")
    per_iter = num_envs * PPOCfg.num_steps / rate
    print(f"=> roughly {per_iter:.1f} s of simulation per training iteration, plus the PPO update")


def main():
    parser = argparse.ArgumentParser(
        description="Sim Robot Trainer - teach a simulated biped to walk and climb stairs with PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a policy")
    p.add_argument("--flat", action="store_true", help="flat ground only (no stairs)")
    p.add_argument("--num_envs", type=int, default=PPOCfg.num_envs, help="robots simulated in parallel")
    p.add_argument("--iterations", type=int, default=PPOCfg.max_iterations, help="PPO iterations to run")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu (neural networks)")
    p.add_argument("--threads", type=int, default=None, help="physics threads (default: all cores)")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="continue from a checkpoint (no path = the newest model)")
    p.add_argument("--name", default=None, help="run name (default: timestamp)")
    p.add_argument("--seed", type=int, default=0)

    def add_scene_args(p):
        p.add_argument("--model", default=None, help="checkpoint (default: newest, else pretrained)")
        p.add_argument("--terrain", default="stairs_up", choices=["flat", "stairs_up", "stairs_down"])
        p.add_argument("--level", type=int, default=5, help="stairs difficulty level 0-9")
        p.add_argument("--speed", type=float, default=0.6, help="commanded forward speed [m/s]")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("play", help="watch a trained policy in the MuJoCo viewer")
    add_scene_args(p)

    p = sub.add_parser("record", help="render an episode to a video file")
    add_scene_args(p)
    p.add_argument("--output", default="walk.mp4", help="output file (.mp4 or .gif)")
    p.add_argument("--seconds", type=float, default=12.0)

    p = sub.add_parser("eval", help="measure success rates of a trained policy")
    p.add_argument("--model", default=None)
    p.add_argument("--num_envs", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)

    p = sub.add_parser("bench", help="measure simulation speed on this machine")
    p.add_argument("--num_envs", type=int, default=PPOCfg.num_envs)
    p.add_argument("--threads", type=int, default=None)

    p = sub.add_parser("list", help="list saved models")
    p.add_argument("--models_dir", default=MODEL_SAVE_PATH)

    args = parser.parse_args()

    if args.command == "train":
        from train import train
        resume = args.resume
        if resume == "latest":
            resume = resolve_model(None)
        elif resume and not os.path.exists(resume):
            sys.exit(f"Model not found: {resume}")
        train(num_envs=args.num_envs, max_iterations=args.iterations, flat_only=args.flat,
              device=args.device, num_threads=args.threads, resume=resume, run_name=args.name, seed=args.seed)
    elif args.command == "play":
        from play import play
        play(resolve_model(args.model), args.terrain, args.level, args.speed, seed=args.seed)
    elif args.command == "record":
        from play import record
        record(resolve_model(args.model), args.output, args.terrain, args.level, args.speed,
               seconds=args.seconds, seed=args.seed)
    elif args.command == "eval":
        from play import evaluate
        evaluate(resolve_model(args.model), args.num_envs, args.device, args.threads)
    elif args.command == "bench":
        bench(args.num_envs, args.threads)
    elif args.command == "list":
        list_models(args.models_dir)


if __name__ == "__main__":
    main()
