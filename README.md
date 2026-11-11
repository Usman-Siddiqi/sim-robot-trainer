# Sim Robot Trainer

Teaches a simulated humanoid two things with reinforcement learning (PPO):

1. **Hike** - follow winding mountain trails: irregular rock steps, stepping stones with
   gaps, narrow ridges, boulders. Fall off the side and it is a one metre drop.
2. **Sort** - pick up crates of different weights and carry each one to the table of its
   colour.

Thousands of robots are simulated in parallel with [MuJoCo](https://mujoco.org/), curricula
make the tasks harder as the robots get better, the physics is randomised so the controller
cannot overfit to one simulator setup, and a from-scratch PPO implementation in PyTorch
learns the controllers.

**Status**

| Task | State | Bundled checkpoint |
| --- | --- | --- |
| hike | trained (2,400 iterations) and evaluated, see [results](#hike-results) | `pretrained/hike.pt` |
| sort | **training in progress, it does not sort yet.** After 450 iterations the robot walks to the table, hooks its palms under the rim of the crate and lifts it; carrying it to the right table and putting it down is what is left to learn. See [continuing the sorting training](#continuing-the-sorting-training). | `pretrained/sort.pt` |

A 2.5 minute video of the hiking policy on seven trails - four training trails, a trail
layout and switchbacks it never trained on, and a hike with a heavy backpack while being
shoved: [docs/hike.mp4](docs/hike.mp4).

Built as a reinforcement learning practice project, with the main goal of understanding
how the robot model, environment design, rewards and training setup decide whether a robot
learns anything at all.

## What is in it

- **20-DoF humanoid** (`robot/humanoid.xml`): two legs (hip yaw / roll / pitch, knee, ankle
  pitch / roll) and two arms (shoulder pitch / roll / yaw, elbow) ending in flat palms,
  28.5 kg, PD position servos with torque limits. The arms are free to swing for balance
  when hiking and carry the crates when sorting.
- **Hiking trails** (`terrain.py`): procedurally generated, raised, winding paths built from
  rock columns, assembled from sections (rock steps, stepping stones, ridges, boulders, ...)
  in 10 difficulty levels. A curriculum moves every robot to harder or easier trails
  depending on how it did.
- **Sorting room** (`env_sort.py`): a pickup table, three coloured drop-off tables (which is
  which changes every episode) and crates of 0.5 - 6 kg in five sizes. The robot is not told
  the weight of a crate.
- **Generalisation**: training randomises friction, payload, motor strength, sensor noise
  and shoves; `python main.py eval` then tests on trail sections, layouts and physics that
  were never used in training.
- **Fast**: physics for all robots is stepped by MuJoCo's multi-threaded rollout on every
  CPU core, the neural networks run on the GPU.
- **Tools**: interactive viewer, video export, evaluation tables, TensorBoard logging,
  resumable checkpoints.

Tuned for a Ryzen 7 8845HS + RTX 4050 laptop, but it runs on anything (a GPU is optional).

## Setup

Python 3.10+ (developed and tested with Python 3.13, MuJoCo 4.0 and PyTorch 2.15).

```bash
git clone https://github.com/Usman-Siddiqi/sim-robot-trainer.git
cd sim-robot-trainer
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install PyTorch **with CUDA first** (a plain `pip install torch` on Windows installs the
CPU-only build and the RTX 4050 would sit idle):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then the rest:

```bash
pip install -r requirements.txt
```

Check that everything is in place (prints the device PyTorch uses and the simulation speed):

```bash
python main.py bench --task hike
```

## Usage

Every command takes `--task hike` (the default) or `--task sort`.

Watch the bundled policies in the MuJoCo viewer:

```bash
python main.py play --task hike --terrain trail --level 7
```

```bash
python main.py play --task sort
```

Hiking trails to choose from: `flat`, `rocky`, `climb`, `stones`, `ridge`, `trail` (the
training trails) and `logs`, `terraces`, `rubble`, `staircase`, `switchbacks`, `new_trail`
(never used in training). Keys in the viewer: BACKSPACE restarts, UP / DOWN change the
walking speed.

Measure success rates, including conditions the policy never trained in:

```bash
python main.py eval --task hike
```

Render a video without opening a window:

```bash
python main.py record --task hike --terrain stones --level 6 --output stones.mp4
```

Train. Checkpoints go to `models/<run name>/` every 50 iterations, Ctrl+C saves and stops,
`--resume` continues from the newest model of that task (or from the bundled one if you
have none):

```bash
python main.py train --task hike --resume
```

```bash
python main.py train --task sort --resume
```

Training curves:

```bash
tensorboard --logdir logs
```

### Continuing the sorting training

The bundled `pretrained/sort.pt` is a full checkpoint (network, optimizer, curriculum
state) of an unfinished run. To carry on from it:

```bash
python main.py train --task sort --resume pretrained/sort.pt --iterations 3000 --name sort_laptop
```

What to watch in the console (the same numbers are in TensorBoard under `episode/`):

| Column | Meaning | Where the bundled checkpoint is |
| --- | --- | --- |
| `stage` | furthest stage reached per episode, averaged: 0 fetch, 1 lift, 2 carry, 3 lower, 4 release | about 1 |
| `sorted` | crates sorted per 30 s episode | 0 |
| `kg` | heaviest crate in use; one more weight is added whenever `stage` reaches 1.7 | 1.5 |
| `guided` | share of robots that are told where to walk, see below; goes down once `sorted` reaches 1 | 1.00 |
| `drop` / `wrong` | crates dropped / put on the wrong table per episode | 0.6 / 0 |

The run is finished when `guided` is 0.00, `kg` is 6.00 and `sorted` has stopped rising.
How long that takes is not known yet: nobody has run it to the end. On the 4-core CPU it
was developed on it ran at about 3,500 steps/s (16 s per iteration); the laptop should be
several times faster. If `stage` stays below 1.7 for many hundreds of iterations, the
rewards in `SortCfg` (config.py) are the place to experiment - the section
[what was learned building this](#what-was-learned-building-this) lists what has been
tried.

To start the sorting training from the beginning instead (it starts from a hiking
policy - your newest one, or the bundled one):

```bash
python main.py train --task sort --name sort_from_hiker
```

## Hike results

`pretrained/hike.pt` after 2,400 iterations (49 million simulation steps), measured with
`python main.py eval --task hike --num_envs 1000`: one episode per robot, exploration noise
off, success = reaching the end of the trail within 30 s.

**Training trails**, nominal robot (success per difficulty level, L9 is the hardest):

| Trail | Success | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat (no trail: not falling) | 100% | | | | | | | | | | |
| rocky (bumpy paving, rock steps) | 100% | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| climb (steps up to 20 cm) | 100% | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| ridge (down to 50 cm wide, boulders) | 99% | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 92 |
| trail (everything mixed) | 81% | 100 | 100 | 100 | 100 | 90 | 83 | 90 | 100 | 50 | 0 |
| stones (stepping stones) | 62% | 100 | 100 | 100 | 85 | 77 | 42 | 100 | 0 | 0 | 0 |

**Trails it never trained on** (sections that do not exist in training, or lie outside the
training ranges, on layouts generated with a different seed):

| Trail | Success |
| --- | --- |
| rubble (smaller, taller bumps) | 99% |
| terraces (left and right half of the trail at different heights) | 93% |
| switchbacks (swings 60% further left / right) | 90% |
| new_trail (training sections and the new ones mixed) | 81% |
| logs (logs across the trail, up to 20 cm) | 69% |
| staircase (regular stairs up to 22 cm) | 65% |

**Physics it never trained with**, on the training trails:

| Condition | Success |
| --- | --- |
| slippery (friction 0.4; training 0.5 - 1.25) | 88% |
| heavy backpack (+6 kg; training up to +3 kg) | 87% |
| weak motors (80%; training 90 - 110%) | 87% |
| hard shoves (0.8 m/s every 3 s; training 0.5 m/s every 5 s) | 83% |
| all of that at once (milder) | 78% |

The limits are where the table says they are: stepping stones with 25 cm gaps (levels
7 - 9; levels 3 - 6 have 12.5 cm gaps) are not solved, and that drags down the hardest
mixed trails and every "unseen physics" row, where `stones` is by far the weakest trail.
The stepping stone level was still rising when training stopped, so more iterations should
help:

```bash
python main.py train --task hike --resume pretrained/hike.pt --iterations 1000 --name hike_more
```

## How it works

```
main.py        command line interface
config.py      every tunable number: robot, both tasks, randomisation, rewards, PPO
robot/         humanoid.xml - the MuJoCo model
terrain.py     hiking trails: column heights -> collision boxes + height lookup + path
env_base.py    batched simulation, robot bookkeeping, shared rewards
env_hike.py    task 1: commands, height scan, curriculum
env_sort.py    task 2: the room, the crates, stages, guidance
agent.py       actor-critic networks, PPO, checkpoints, policy transfer
train.py       the training loop
play.py        viewer, video recorder, evaluation
showcase.py    records the captioned demo video
```

**Simulation.** All robots are stepped together by `mujoco.rollout` on a C++ thread pool;
observations and rewards are computed for the whole batch in numpy. Every robot lives in
its own copy of one scene (one trail tile, or the sorting room), so robots never meet. A
trail tile is one small MuJoCo model; a single world containing all 60 trails would be
several times slower to step.

**Control.** The policy runs at 50 Hz and outputs 20 joint position targets (offsets from a
default pose) that PD servos track at 200 Hz.

**Hiking.** A navigation layer turns "where does the trail go" into a velocity command
(forward, sideways, turning rate): walk towards the point 0.75 m ahead on the trail and face
the way the trail runs. The policy sees that command, its own motion and joints, a gait
clock and a 16 x 9 height scan of the ground ahead (218 numbers), and has to realise the
command without falling off.

**Sorting.** The reward is a progress score, `5 x crates sorted + stage + progress within
the stage`, with the stages fetch, lift, carry, lower, release - finishing a stage always
pays more than lingering in it. Which stage a robot is in is worked out from the physical
state (where the palms and the crate are, how much of the crate's weight still rests on
the table).

The sorting policy is not trained from nothing: it starts as a copy of the hiking policy
(its observation begins with the hiking observation, the weights of the 35 extra inputs
start at zero), so it can walk from the first iteration. At first a "guide" - a few lines
of code that know the rules - gives it velocity commands to the crate and then to the
right table. Once robots sort crates that way the guide is withdrawn from more and more
robots, which then only see the colour of the crate and where the red, the green and the
blue table are. A finished policy gets no commands at all and does the sorting itself.

**Learning.** PPO with generalised advantage estimation, a clipped value loss, observation
normalisation and a learning rate that adapts to keep the policy change per update constant
(`agent.py`, about 300 lines).

## What was learned building this

Things that did not work, in case you run into the same walls:

- **A reward the robot can collect by doing nothing.** Twice the hiking robot learned to
  stand still: whenever it faced the wrong way it was asked for a slow walk plus a sharp
  turn, and the speed reward is so forgiving for slow commands that not moving cost almost
  nothing. Commands now keep a real walking speed while turning.
- **Steering that asks for the wrong skill.** On zigzag stepping stones the navigation layer
  used to turn the robot towards every next stone. It now keeps facing along the trail and
  asks for a sideways step; stepping stones went from level 2 to level 6.
- **Exploration noise that grows.** With arms added, the entropy bonus kept inflating the
  noise of joints that hardly affect the reward (ankle roll, shoulder roll) until precise
  foot placement was impossible. The bonus was lowered and the noise capped.
- **Sorting from scratch.** One policy learning to stand, walk, grasp and carry all at once
  was still falling over after 150 iterations. Starting from the hiking policy fixed that.
- **Squeezing a smooth box between two flat palms.** Five reward variants got the robot to
  reach and squeeze, never to lift: holding a box by friction alone needs well coordinated
  shoulder torques close to their limits, and the reward is flat until the box actually
  leaves the table. The boxes became crates with a rim the palms hook under, and the lift
  is rewarded by the share of the weight the hands already carry.
- **Penalties that teach avoidance.** A penalty for dropping a crate taught the robots to
  keep their hands away from crates. A dropped crate is now simply replaced.

## Hardware notes (Ryzen 7 8845HS + RTX 4050)

- Physics is CPU bound: more cores, more robots per second. `python main.py bench` shows
  what your machine does. Plug the laptop in and use the performance power profile.
- The networks are small; the GPU mainly makes the PPO update fast. If PyTorch reports
  `device: cpu`, the CPU-only build of torch is installed - reinstall with the CUDA command
  from the setup section.
- 2,048 robots is the default. `--num_envs 4096` gives smoother learning if the CPU keeps
  up; halve it if the machine is short of memory.

## License

See [LICENSE](LICENSE).
