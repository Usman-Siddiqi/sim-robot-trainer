# Sim Robot Trainer

Teaches a simulated humanoid two things with reinforcement learning (PPO):

1. **Hike** - follow winding mountain trails: slopes up, down and sideways, rock steps,
   stepping stones with gaps, ridges, a balance beam, a plank bridge with hand rails, roots
   and logs to step over, walls, gates and rocks to get around. Fall off the side and it is
   a one metre drop.
2. **Sort** - pick up crates of 0.5 - 6 kg (the robot is not told the weight) and carry each
   one to the table of its colour.

Thousands of robots are simulated in parallel with [MuJoCo](https://mujoco.org/), curricula
make the tasks harder as the robots get better, the physics is randomised so the controller
cannot overfit to one simulator setup, and a from-scratch PPO implementation in PyTorch
learns the controllers.

**Status**

| Task | State | Bundled checkpoint |
| --- | --- | --- |
| sort | **works without help**: 4.97 crates sorted per 30 s episode, 97 % of robots deliver, none to the wrong table - see [sort results](#sort-results) | `pretrained/sort.pt` |
| hike | trained on the hard trails (3,200 iterations); the arms still stay clamped to the body - a fix is in `config.py` / `env_hike.py` but not trained yet | `pretrained/hike.pt` |

Videos: [docs/sort.mp4](docs/sort.mp4) (sorting, including crates heavier than any in
training) and [docs/hike.mp4](docs/hike.mp4) (an earlier hiking policy, before the trails got
harder).

Built as a reinforcement learning practice project, with the main goal of understanding
how the robot model, environment design, rewards and training setup decide whether a robot
learns anything at all.

## What is in it

- **20-DoF humanoid** (`robot/humanoid.xml`): two legs (hip yaw / roll / pitch, knee, ankle
  pitch / roll) and two arms (shoulder pitch / roll / yaw, elbow) ending in flat palms,
  28.5 kg, PD position servos with torque limits. The arms are free to swing for balance
  when hiking and carry the crates when sorting.
- **Hiking trails** (`terrain.py`): procedurally generated, raised, winding paths built from
  rock columns and tilted slabs, assembled from 22 kinds of section (ramps, banked and
  V-shaped slopes, rock steps, stepping stones, ridges, a beam, a plank bridge with rails,
  roots, logs, loose stones, canyons, gates, rocks in the way, a ledge, ...) in 10
  difficulty levels. A curriculum moves every robot to harder or easier trails depending
  on how it did, and more robots to the kinds of trail they fail on.
- **Sorting room** (`env_sort.py`): a pickup table, three coloured drop-off tables (which is
  which changes every episode) and crates of 0.5 - 6 kg in five sizes. The robot is not told
  the weight of a crate.
- **Generalisation**: training randomises friction, payload, motor strength, sensor noise
  and shoves; `python main.py eval` then tests on trail sections, layouts and physics that
  were never used in training.
- **Fast**: physics for all robots is stepped by MuJoCo's multi-threaded rollout, split over
  8 processes so that observations and rewards are computed in parallel too; the networks
  run on the GPU. About 55,000 simulation steps per second on the laptop (a 1,000-iteration
  run takes half an hour).
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

Hiking trails to choose from: `flat`, `rocky`, `climb`, `slopes`, `stones`, `ridge`, `walls`,
`hazards`, `trail` (the training trails) and `terraces`, `rubble`, `staircase`, `steep`,
`debris`, `switchbacks`, `new_trail` (never used in training). Keys in the viewer: BACKSPACE restarts, UP / DOWN change the
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

Follow the training that is running (speed, progress, minutes left; Ctrl+C stops watching,
not the training). You can open the viewer at the same time - it loads the newest checkpoint:

```bash
python main.py watch
```

Training curves:

```bash
tensorboard --logdir logs
```

### How the sorting policy was trained

1. **From the hiker.** The sorting policy starts as a copy of a hiking policy, so it can walk
   from the first iteration (`python main.py train --task sort` does this by itself).
2. **With a guide.** At first a few lines of code (the "guide") tell it where to walk: to the
   crate, then to the table of the crate's colour. It learns to grip, lift, carry, put down
   and let go. Heavier crates are added only once it actually delivers.
3. **Without the guide, by imitation.** The guided policy teaches a copy of itself that does
   not see the guide's commands; the copy works out where to go from the crate's colour and
   where the coloured tables are:

```bash
python main.py distill --teacher models/<run>/model_<n>.pt
```

The console columns while training: `stage` (furthest stage per episode: 0 fetch, 1 lift,
2 carry, 3 lower, 4 release), `sorted` (crates per 30 s episode), `kg` (heaviest crate in
use), `guided` (share of robots that get the guide's commands), `alone` (crates sorted by
the robots without them).

## Sort results

`pretrained/sort.pt`, measured with `python main.py eval --task sort --num_envs 600`: no
guide, exploration noise off, one 30 s episode per robot.

| Condition | Crates sorted | Robots that sort at least one | Robots that fell | Wrong table |
| --- | --- | --- | --- | --- |
| training conditions (crates 0.5 - 6 kg) | 4.97 | 97 % | 1 % | 0.00 |
| heavier crates (6.5 - 8 kg, never in training) | 3.18 | 88 % | 4 % | 0.00 |
| very light crates (0.3 - 1 kg) | 4.72 | 96 % | 2 % | 0.00 |
| heavy backpack (+6 kg) | 4.37 | 96 % | 4 % | 0.00 |
| shoves (0.6 m/s every 4 s) | 4.89 | 98 % | 2 % | 0.00 |
| weak motors (80 %) | 2.34 | 78 % | 10 % | 0.01 |
| slippery floor (friction 0.4) | 1.84 | 82 % | 51 % | 0.00 |

The first crate is delivered by 86 - 100 % of the robots at every weight from 0.5 to 6 kg,
on average 4.2 s into the episode. The weak spot is a slippery floor.

## Hike results

*These numbers are for an earlier policy, before slopes, hazards, walls, the beam and the
bridge were added; the bundled `pretrained/hike.pt` is the newer one and has not been
measured the same way yet.*

The policy after 2,400 iterations (49 million simulation steps), measured with
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
default pose) that PD servos track; physics takes two 10 ms steps per decision.

**Hiking.** A navigation layer turns "where does the trail go" into a velocity command
(forward, sideways, turning rate): walk towards the point 0.75 m ahead on the trail and face
the way the trail runs. The policy sees that command, its own motion and joints, a gait
clock, a 16 x 9 height scan of the ground ahead and the contact force on its palms (220
numbers), and has to realise the command without falling off. The arms get a small reward
for swinging against the legs and are otherwise free.

**Sorting.** The reward is a progress score over the stages, with the stages fetch, lift, carry, lower, release - finishing a stage always
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
- **Rewards that pay for a half-solution.** Hooking the palms under one end of the crate
  tipped it like a lid and counted as lifting; then holding a crate just above the table
  paid nearly as much as carrying it. Only an upright lift counts now, and carrying is
  worth three times as much as holding.
- **Curricula that run ahead.** Heavier crates were added whenever robots briefly reached
  the carry stage (2 kg to 6 kg in 300 iterations), and the guide was withdrawn from 90 %
  of the robots in 200 iterations; both times deliveries collapsed. Both now move in small
  steps, need real deliveries, and the guide comes back when deliveries drop. Removing the
  guide finally worked by imitation (distillation), not by trial and error.
- **A robot that braces against itself.** The hiker learned to press its palms against its
  thighs and its forearms against its trunk - the arm motors then help the legs. Penalties
  on the palms alone just moved the bracing to the elbows.

## Hardware notes (Ryzen 7 8845HS + RTX 4050)

- Physics is CPU bound: more cores, more robots per second. `python main.py bench` shows
  what your machine does. Plug the laptop in and use the performance power profile.
- The networks are small; the GPU mainly makes the PPO update fast. If PyTorch reports
  `device: cpu`, the CPU-only build of torch is installed - reinstall with the CUDA command
  from the setup section.
- 4,096 robots is the default; halve it (`--num_envs 2048`) if the machine is short of
  memory.
- **Plug the laptop in.** On battery Windows switches the RTX 4050 off and throttles the CPU:
  training drops from about 55,000 to under 10,000 steps per second.

## License

See [LICENSE](LICENSE).
