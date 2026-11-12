"""
distill.py

Author: Usman Siddiqi

Takes the guide away from a sorting policy by imitation.

A sorting policy first learns with a guide that tells it where to walk (to the
crate, then to the table of the crate's colour - see "Guidance" in SortCfg).
Everything the guide knows can be worked out from what the robot sees anyway:
the colour of the crate and where the three coloured tables are. But finding
that out by trial and error is slow, because a robot without commands at first
just stands there.

So the guided policy (the "teacher") teaches a copy of itself that does not get
the commands (the "student"): both look at the same situation, the teacher with
the guide's commands and the student without, and the student is trained to move
its joints the way the teacher does. The student also does the acting, so it
learns to recover from the situations its own mistakes get it into.

    python main.py distill                              # newest sorting model -> models/sort_distilled/model_final.pt
    python main.py distill --teacher models/sort_v8/model_4600.pt --iterations 400

Afterwards the student can be trained further without a guide:

    python main.py train --task sort --resume models/sort_distilled/model_final.pt
"""

import copy
import os
import time

import numpy as np
import torch

from agent import get_device, load_policy, print_device_info, save_checkpoint
from config import MODEL_SAVE_PATH, PPOCfg, SortCfg

COMMAND = slice(71, 74)         # where the guide's velocity command sits in the observation


def distill(teacher_path, output=None, num_envs=PPOCfg.num_envs, iterations=300, device="auto", num_threads=None,
            seed=0, teacher_share=40, learning_rate=5.0e-4):
    """
    Args:
        teacher_path: checkpoint of a sorting policy that works with the guide
        output: where to save the student (default: models/sort_distilled/model_final.pt)
        iterations: rounds of 24 steps per robot, each followed by training on what was seen
        teacher_share: for this many iterations some of the robots are still moved by the teacher (fewer
                       every iteration), so that the student sees what a whole delivery looks like from the start
    Returns:
        path of the student's checkpoint
    """
    from train import make_env

    device = get_device(device)
    print_device_info(device)
    teacher, ckpt = load_policy(teacher_path, device)
    if ckpt.get("task") != "sort":
        raise ValueError(f"{teacher_path} is not a sorting policy")
    teacher.eval()
    student = copy.deepcopy(teacher)
    optimizer = torch.optim.Adam(student.actor.parameters(), lr=learning_rate)

    env = make_env("sort", num_envs=num_envs, num_threads=num_threads, seed=seed)
    num_envs = env.num_envs
    # the teacher always sees the guide; all box weights from the start
    env.load_curriculum({"guide": 1.0, "mass_level": SortCfg.num_box_masses}, np.random.default_rng(seed))
    output = output or os.path.join(MODEL_SAVE_PATH, "sort_distilled", "model_final.pt")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    def save(path):
        save_checkpoint(path, student, None, ckpt.get("iteration", 0), {
            "task": "sort", "obs_layout": env.obs_layout,
            "curriculum": {"guide": 0.0, "mass_level": SortCfg.num_box_masses}})

    print(f"Teacher {teacher_path}: {num_envs} robots, {iterations} iterations")
    obs = torch.from_numpy(env.reset()).to(device)
    steps_per_episode = env.max_episode_length
    start = time.time()
    try:
        for iteration in range(1, iterations + 1):
            share = max(0.0, 1.0 - iteration / max(teacher_share, 1)) if teacher_share else 0.0
            seen, wanted = [], []
            delivered, own_steps = 0.0, 0.0
            for _ in range(PPOCfg.num_steps):
                with torch.no_grad():
                    blind = obs.clone()
                    blind[:, COMMAND] = 0.0                         # what the student gets to see
                    target = teacher.act_inference(obs)             # what the teacher would do here
                    action = student.act_inference(blind)
                    by_teacher = torch.rand(num_envs, device=device) < share
                    action = torch.where(by_teacher[:, None], target, action)
                    action = action + 0.1 * torch.randn_like(action)    # a little noise: more varied situations
                seen.append(blind)
                wanted.append(target)
                obs_np, _, _, info = env.step(action.cpu().numpy())
                obs = torch.from_numpy(obs_np).to(device)
                own = ~by_teacher.cpu().numpy()
                delivered += float(info["sorted"][own].sum())
                own_steps += float(own.sum())

            seen, wanted = torch.cat(seen), torch.cat(wanted)
            losses = []
            for _ in range(4):
                order = torch.randperm(seen.shape[0], device=device)
                for batch in order.split(seen.shape[0] // 6):
                    loss = ((student.actor(student.normalizer(seen[batch])) - wanted[batch]) ** 2).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

            if iteration % 10 == 0 or iteration == 1:
                per_episode = delivered / max(own_steps, 1.0) * steps_per_episode
                eta = (time.time() - start) / iteration * (iterations - iteration) / 60
                print(f"it {iteration:4d}/{iterations} | difference to the teacher {np.mean(losses):.4f} | moved by the teacher "
                      f"{share:4.0%} | student alone sorts {per_episode:4.2f} boxes per episode | eta {eta:4.1f} min", flush=True)
            if iteration % 100 == 0:
                save(output)
    except KeyboardInterrupt:
        print("\nInterrupted - saving the student as it is...")
    save(output)
    env.close()
    print(f"Saved the student: {output}")
    print(f"Test it:    python main.py eval --task sort --model {output}")
    print(f"Watch it:   python main.py play --task sort --model {output}")
    return output


if __name__ == "__main__":
    import sys
    distill(sys.argv[1], iterations=int(sys.argv[2]) if len(sys.argv) > 2 else 300)
