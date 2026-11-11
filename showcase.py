"""
showcase.py

Author: Usman Siddiqi

Records a captioned video of a policy in several scenarios: terrains it was
trained on, then terrains and physics it never saw in training.

    python showcase.py                                  # pretrained policy -> docs/showcase.mp4
    python showcase.py --model models/<run>/model_final.pt --output my_run.mp4

Needs: pip install imageio imageio-ffmpeg   (Pillow comes with imageio)
"""
import argparse
import os

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from agent import load_policy
from config import EvalCfg, TerrainCfg
from environment import BipedEnv
from play import _tracking_camera

W, H, FPS = 960, 544, 25


def font(size, bold=False):
    """DejaVu (Linux), Arial (Windows) or Helvetica (macOS), else Pillow's built-in font."""
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Helvetica-Bold.ttc"] if bold else
             ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "Helvetica.ttc"])
    for name in names:
        for path in (name, f"/usr/share/fonts/truetype/dejavu/{name}", f"/Library/Fonts/{name}",
                     f"/System/Library/Fonts/{name}"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


f_title, f_sub, f_tag, f_big = font(30, True), font(20), font(16, True), font(44, True)
ROWS = {r["name"]: r for r in TerrainCfg.rows + EvalCfg.unseen_terrains}

SCENARIOS = [
    dict(title="Walking + turning on command", sub="flat ground, 0.7 m/s, heading changes every 3 s",
         tag="TRAINED", terrain="flat", level=0, speed=0.7, seconds=10, turns=True),
    dict(title="Climbing stairs", sub="hardest level: 16 cm steps",
         tag="TRAINED", terrain="stairs_up", level=9, speed=0.6, seconds=14),
    dict(title="Walking down stairs", sub="hardest level: 16 cm steps",
         tag="TRAINED", terrain="stairs_down", level=9, speed=0.6, seconds=14),
    dict(title="Irregular stairs", sub="every step a different height and depth",
         tag="NEVER SEEN IN TRAINING", terrain="irregular_stairs_up", level=9, speed=0.6, seconds=14),
    dict(title="Scattered blocks", sub="random blocks up to 13 cm",
         tag="NEVER SEEN IN TRAINING", terrain="scattered_blocks", level=9, speed=0.6, seconds=12),
    dict(title="Rolling hills", sub="waves up to 35 cm peak to peak",
         tag="NEVER SEEN IN TRAINING", terrain="waves", level=9, speed=0.6, seconds=12),
    dict(title="Getting shoved", sub="0.8 m/s kick every 3 s (training: 0.5 m/s every 5 s)",
         tag="NEVER SEEN IN TRAINING", terrain="flat", level=0, speed=0.6, seconds=12,
         push=dict(max_push_vel=0.8, push_interval_s=3.0)),
    dict(title="Stairs while overloaded", sub="+6 kg payload and 80 % motor strength",
         tag="NEVER SEEN IN TRAINING", terrain="stairs_up", level=9, speed=0.6, seconds=14,
         physics=dict(payload=6.0, motor_strength=0.8)),
]


def caption(img, sc, extra=None):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, W, 84], fill=(15, 18, 24, 175))
    d.text((20, 10), sc["title"], font=f_title, fill=(255, 255, 255))
    d.text((20, 50), sc["sub"], font=f_sub, fill=(205, 212, 222))
    color = (60, 160, 90, 235) if sc["tag"] == "TRAINED" else (210, 110, 40, 235)
    tw = d.textlength(sc["tag"], font=f_tag)
    d.rounded_rectangle([W - tw - 40, 26, W - 16, 58], 8, fill=color)
    d.text((W - tw - 28, 33), sc["tag"], font=f_tag, fill=(255, 255, 255))
    if extra:
        tw = d.textlength(extra, font=f_big)
        d.text(((W - tw) / 2, H - 90), extra, font=f_big, fill=(255, 220, 60), stroke_width=3, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def title_card(lines, seconds=2.4):
    im = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(im)
    y = H / 2 - 40 * len(lines) / 2
    for text, font, color in lines:
        tw = d.textlength(text, font=font)
        d.text(((W - tw) / 2, y), text, font=font, fill=color)
        y += 52
    return [np.asarray(im)] * int(seconds * FPS)


def record(sc, seed=0):
    flat = dict(TerrainCfg.flat_rows[0], weight=0.0)
    rows = [dict(ROWS["flat"], weight=1.0)] if sc["terrain"] == "flat" else [flat, dict(ROWS[sc["terrain"]], weight=1.0)]
    env = BipedEnv(num_envs=1, flat_only=sc["terrain"] == "flat", num_threads=1, seed=seed,
                   terrain_rows=None if sc["terrain"] == "flat" else rows, terrain_seed=seed,
                   curriculum=False, fixed_level=sc["level"], randomize=False,
                   physics=sc.get("physics"), push=sc.get("push"),
                   fixed_speed=sc["speed"], resample_commands=False)
    data = mujoco.MjData(env.model)
    renderer = mujoco.Renderer(env.model, H, W)
    cam = _tracking_camera(env.model)
    obs = env.reset()
    every = int(round(1.0 / FPS / env.dt))
    frames, outcome, flash = [], "time is up", 0
    turn_every = int(3.0 / env.dt)
    for step in range(int(sc["seconds"] / env.dt)):
        if sc.get("turns") and step > 0 and step % turn_every == 0:
            env.heading_target[0] += np.radians(70) * (1 if (step // turn_every) % 2 else -1)
        actions = policy.act_inference(torch.from_numpy(obs))
        obs, _, done, info = env.step(actions.numpy())
        if sc.get("push") and env.common_step % env.push_interval == 0:
            flash = int(0.8 / env.dt)
        if done[0]:
            outcome = "fell" if info["fell"][0] else ("reached the goal" if info["reached_goal"][0] else outcome)
            break
        if step % every == 0:
            env.write_data(0, data)
            renderer.update_scene(data, camera=cam)
            frames.append(caption(renderer.render(), sc, "SHOVE!" if flash > 0 else None))
        flash -= 1
    renderer.close()
    env.close()
    frames += [frames[-1]] * int(0.6 * FPS)        # hold the last frame briefly
    return frames, outcome


def main():
    parser = argparse.ArgumentParser(description="Record a captioned showcase video of a trained policy")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", "biped.pt"))
    parser.add_argument("--output", default=os.path.join("docs", "showcase.mp4"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    global policy
    policy, _ = load_policy(args.model, torch.device("cpu"))
    policy.eval()

    all_frames = title_card([("Sim Robot Trainer", font(46, True), (255, 255, 255)),
                             ("12-DoF biped, PPO policy trained in MuJoCo", f_sub, (200, 208, 220)),
                             ("one policy, no retuning between clips", f_sub, (200, 208, 220))])
    for i, sc in enumerate(SCENARIOS):
        if i == 3:
            all_frames += title_card([("Now: things it never trained on", font(38, True), (240, 150, 70)),
                                      ("new terrain shapes and physics outside the training ranges", f_sub, (200, 208, 220))],
                                     seconds=2.0)
        frames, outcome = record(sc, args.seed)
        print(f"{sc['title']:<32} {outcome:<17} {len(frames) / FPS:5.1f} s", flush=True)
        all_frames += frames

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    imageio.mimsave(args.output, all_frames, fps=FPS, codec="libx264", quality=7, macro_block_size=16,
                    ffmpeg_params=["-movflags", "+faststart"])
    print(f"saved {args.output}: {len(all_frames) / FPS:.1f} s, {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
