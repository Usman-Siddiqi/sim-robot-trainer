"""
showcase.py

Author: Usman Siddiqi

Records one captioned video of the trained policies in several scenarios:
hiking trails (training trails, then trails and conditions the policy never
trained on) and box sorting (normal boxes, then boxes heavier than any in
training).

    python showcase.py                              # bundled policies -> docs/showcase.mp4
    python showcase.py --only hike --output hike.mp4
    python showcase.py --hike models/my_run/model_final.pt

Needs: pip install imageio imageio-ffmpeg pillow
On a machine without a display set MUJOCO_GL=egl first.
"""

import argparse
import os

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import PRETRAINED
from play import Scene, load, make_scene, render_episode

W, H, FPS = 960, 544, 25

HIKE = [
    dict(title="Rock steps", sub="irregular steps up and down, up to 20 cm", tag="TRAINED",
         terrain="climb", level=9, speed=0.6),
    dict(title="Stepping stones", sub="zigzag stones with 25 cm gaps - miss one and it is a 1 m drop", tag="TRAINED",
         terrain="stones", level=9, speed=0.6),
    dict(title="Ridge and boulders", sub="50 cm wide ridge", tag="TRAINED",
         terrain="ridge", level=9, speed=0.6),
    dict(title="The full hike", sub="everything mixed, hardest level", tag="TRAINED",
         terrain="trail", level=9, speed=0.7),
    dict(title="A trail it has never walked", sub="new layout with logs, terraces and rubble it never trained on",
         tag="NEVER SEEN IN TRAINING", terrain="new_trail", level=8, speed=0.6),
    dict(title="Switchbacks", sub="swings left / right 60 % further than any training trail",
         tag="NEVER SEEN IN TRAINING", terrain="switchbacks", level=8, speed=0.6),
    dict(title="Hiking with a heavy backpack, getting shoved", sub="+5 kg, weak motors, a shove every 4 s",
         tag="NEVER SEEN IN TRAINING", terrain="trail", level=7, speed=0.6,
         physics=dict(payload=5.0, motor_strength=0.85), push=dict(max_push_vel=0.6, push_interval_s=4.0)),
]
SORT = [
    dict(title="Sorting boxes by colour", sub="each box goes to the table of its colour", tag="TRAINED", seed=0),
    dict(title="Sorting boxes by colour", sub="another round: colours and weights are random", tag="TRAINED", seed=5),
    dict(title="Boxes heavier than any in training", sub="6.5 - 8 kg (training: 1.5 - 6 kg)",
         tag="NEVER SEEN IN TRAINING", seed=2, box_mass=(6.5, 8.0)),
]


def _font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"]
    for name in names + ["/usr/share/fonts/truetype/dejavu/" + names[0]]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


F_TITLE, F_SUB, F_TAG, F_BIG = _font(30, True), _font(20), _font(16, True), _font(40, True)


def caption(frame, sc, note=None, flash=None):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, W, 84], fill=(15, 18, 24, 175))
    d.text((20, 10), sc["title"], font=F_TITLE, fill=(255, 255, 255))
    d.text((20, 50), sc["sub"], font=F_SUB, fill=(205, 212, 222))
    color = (60, 160, 90, 235) if sc["tag"] == "TRAINED" else (210, 110, 40, 235)
    tw = d.textlength(sc["tag"], font=F_TAG)
    d.rounded_rectangle([W - tw - 40, 26, W - 16, 58], 8, fill=color)
    d.text((W - tw - 28, 33), sc["tag"], font=F_TAG, fill=(255, 255, 255))
    if note:
        tw = d.textlength(note, font=F_SUB)
        d.rounded_rectangle([16, H - 48, 36 + tw, H - 14], 8, fill=(15, 18, 24, 175))
        d.text((26, H - 43), note, font=F_SUB, fill=(255, 255, 255))
    if flash:
        tw = d.textlength(flash, font=F_BIG)
        d.text(((W - tw) / 2, H - 110), flash, font=F_BIG, fill=(255, 220, 60), stroke_width=3, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def title_card(lines, seconds=2.4):
    im = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(im)
    y = H / 2 - 26 * len(lines)
    for text, font, color in lines:
        d.text(((W - d.textlength(text, font=font)) / 2, y), text, font=font, fill=color)
        y += 56
    return [np.asarray(im)] * int(seconds * FPS)


def record_scenario(policy, task, device, sc, seed):
    env = make_scene(task, sc.get("terrain", "trail"), sc.get("level", 5), sc.get("speed", 0.6),
                     seed=sc.get("seed", seed), physics=sc.get("physics"), push=sc.get("push"), box_mass=sc.get("box_mass"))
    scene = Scene(policy, env, device)
    state = {"flash": None, "left": 0}

    def on_frame(frame, scene, event):
        if event:
            state["flash"], state["left"] = event.upper() + "!", int(1.2 * FPS)
        if sc.get("push") and env.common_step % env.push_interval < 2:
            state["flash"], state["left"] = "SHOVE!", int(0.8 * FPS)
        state["left"] -= 1
        note = None
        if task == "sort":
            note = f"this box: {env.box_masses[env.box_mass_index[0]]:.1f} kg   sorted so far: {len(scene.sorted_boxes)}"
        return caption(frame, sc, note, state["flash"] if state["left"] > 0 else None)

    frames, events = render_episode(scene, sc.get("seconds", env.episode_length_s), W, H, FPS, on_frame=on_frame)
    env.close()
    if events:                                              # show how it ended on the last frame for a moment
        frames += [caption(frames[-1], dict(sc), None, events[-1].upper() + "!")] * int(1.0 * FPS)
    return frames, events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hike", default=PRETRAINED["hike"], help="hike checkpoint")
    parser.add_argument("--sort", default=PRETRAINED["sort"], help="sort checkpoint")
    parser.add_argument("--only", choices=["hike", "sort"], default=None)
    parser.add_argument("--output", default=os.path.join("docs", "showcase.mp4"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    grey = (200, 208, 220)
    frames = title_card([("Sim Robot Trainer", _font(46, True), (255, 255, 255)),
                         ("20-DoF humanoid, PPO policies trained in MuJoCo", F_SUB, grey)])
    for task, path, scenarios, intro in [
            ("hike", args.hike, HIKE, "Task 1: hiking"), ("sort", args.sort, SORT, "Task 2: sorting boxes by colour")]:
        if args.only not in (None, task):
            continue
        if not os.path.exists(path):
            print(f"no {task} model at {path} - skipped")
            continue
        policy, _, device = load(path)
        frames += title_card([(intro, _font(38, True), (240, 150, 70)), ("one policy for all clips", F_SUB, grey)], 1.8)
        for sc in scenarios:
            clip, events = record_scenario(policy, task, device, sc, args.seed)
            print(f"{sc['title']:<46} {len(clip) / FPS:5.1f} s   {', '.join(events) or 'time is up'}", flush=True)
            frames += clip
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    imageio.mimsave(args.output, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=16,
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    print(f"saved {args.output}: {len(frames) / FPS:.1f} s, {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
