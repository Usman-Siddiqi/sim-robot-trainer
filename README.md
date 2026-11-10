# Sim Robot Trainer

Teaches a simulated two-legged robot to walk and climb stairs with reinforcement learning (PPO).

Thousands of robots are simulated in parallel with [MuJoCo](https://mujoco.org/), a
terrain curriculum moves them from flat ground to taller and taller stairs, and a
from-scratch PPO implementation in PyTorch learns the controller.

![The trained biped climbing 16 cm stairs](docs/stairs_up.gif)

*The bundled pretrained policy climbing the hardest stairs (16 cm steps).*

Built as a reinforcement learning practice project, with the main goal of understanding
how the robot model, environment design, rewards and training setup affect locomotion.

## What it does

- **12-DoF biped** (`robot/biped.xml`): hip yaw / roll / pitch, knee, ankle pitch / roll per leg, ~25 kg, PD position servos with torque limits
- **Tasks**: follow a commanded speed and heading on flat ground, climb stairs up, walk stairs down (step heights 3 cm → 16 cm over 10 curriculum levels)
- **Fast**: physics for all robots is stepped by MuJoCo's multi-threaded rollout on every CPU core, the neural networks run on the GPU
- **Tools**: interactive viewer, video export, evaluation table, TensorBoard logging, resumable checkpoints

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

Install PyTorch **with CUDA first** (a plain `pip install torch` on Windows installs the CPU-only build and the RTX 4050 would sit idle):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then the rest:

```bash
pip install -r requirements.txt
```

Check that everything works and see how fast your machine simulates:

```bash
python main.py bench
```

It should print `device: cuda (NVIDIA GeForce RTX 4050 ...)`. If it says `cpu`, the CUDA build of PyTorch is not installed (training still works, the PPO update is just slower).

## Usage

Watch the bundled pretrained policy climb stairs (no training needed):

```bash
python main.py play --model pretrained/biped_stairs.pt
```

Train the full task (flat ground + stairs up + stairs down):

```bash
python main.py train
```

Train on flat ground only (quick sanity check, walks after a few minutes):

```bash
python main.py train --flat --iterations 400
```

Stop any time with `Ctrl+C` (the policy is saved) and continue later:

```bash
python main.py train --resume
```

Watch your newest model, choosing terrain, stairs level (0-9) and speed:

```bash
python main.py play --terrain stairs_up --level 7 --speed 0.6
```

In the viewer: `UP`/`DOWN` change the speed, `LEFT`/`RIGHT` turn, `BACKSPACE` restarts the episode.

Measure how good a model is (success rate per terrain and stairs level):

```bash
python main.py eval
```

Save a video:

```bash
python main.py record --terrain stairs_up --level 7 --output stairs.mp4
```

List saved models:

```bash
python main.py list
```

Every command has `--help`, e.g. `python main.py train --help`. `play`, `eval` and `record` use the newest model in `models/` unless you pass `--model path/to/model.pt`.

## Training on the 8845HS + RTX 4050

The defaults in `config.py` are chosen for this laptop:

| Setting | Default | Why |
| --- | --- | --- |
| `num_envs` | 2048 robots | enough parallel experience per update while keeping iterations short |
| physics threads | all 16 | MuJoCo rollout thread pool, one robot batch per thread |
| networks | 512-256-128 MLPs on the GPU | one batched forward pass per control step, PPO update on ~12k-sample minibatches |
| physics / control rate | 200 Hz / 50 Hz | 4 physics steps per policy step |

What to expect (one iteration = 2048 robots x 24 steps = 49,152 simulation steps):

| Iterations | Steps | What the robot can do |
| --- | --- | --- |
| ~100 | 5 M | stays upright and starts stepping |
| ~150 | 7 M | walks on flat ground at the commanded speed |
| ~400 | 20 M | climbs 10 cm steps |
| ~500 | 25 M | first successful climbs of the hardest (16 cm) stairs |
| ~800 | 40 M | 99-100 % success on every terrain and level |

These numbers are from an actual run of `python main.py train` with the default settings and seed. That run was done on a 4-core cloud CPU without a GPU (about 4.5 s per iteration, one hour to iteration 800); the 8845HS has four times the threads and much faster cores, and the 4050 takes the PPO update off the CPU, so expect it to be several times faster. `python main.py bench` tells you the simulation speed on your machine.

The default run length is 1500 iterations; the extra iterations after ~800 only make the gait smoother. Stop early with `Ctrl+C` whenever `python main.py eval` looks good enough.

Tips:

- Plug the laptop in and use the performance power profile, otherwise the CPU is throttled hard.
- Simulation speed is CPU-bound. If you want the machine usable while training, pass `--threads 12`.
- `--num_envs 4096` gives smoother learning per iteration at twice the simulation time per iteration; VRAM is not a concern (the policy is tiny).
- Use `--device cpu` to compare, or if CUDA is not available.

## Monitoring

```bash
tensorboard --logdir logs
```

Useful curves: `train/mean_return`, `episode/terrain_level` (curriculum progress), `episode/stairs_success`, `episode/fell` and the individual reward terms under `episode/reward/*`.

The console prints one line every 10 iterations (this one is from the 4-core CPU-only reference run):

```text
it   800/2000 |  11,588 steps/s (sim  1.9s, learn  2.3s) | return   18.41 | ep len  431.1 | fell 0.00 | stairs lvl 6.05 ok 1.00 | std 0.32 | lr 2e-04 | eta  96.8 min
```

`stairs lvl` is the average curriculum level of the stairs robots (0 = 3 cm steps, 9 = 16 cm steps) and `ok` is the fraction of stairs episodes that reached the end of the staircase. Because robots that beat level 9 are sent back to a random level, a policy that has mastered the stairs hovers around level 6 with `ok` near 1.00. Episodes on stairs end as soon as the robot reaches the goal, so `return` and `ep len` are lower than on flat ground - that is expected.

## Pretrained policy

`pretrained/biped_stairs.pt` is the result of `python main.py train` (default settings, seed 0) stopped at iteration 1000 (49 million steps). `python main.py eval --model pretrained/biped_stairs.pt --num_envs 4000`:

| Terrain | Step height | Success | Fell |
| --- | --- | --- | --- |
| flat | - | 100 % | 0 % |
| stairs up, levels 0-9 | 3.0 - 16.0 cm | 99 - 100 % | 0 - 1 % |
| stairs down, levels 0-9 | 3.0 - 16.0 cm | 99 - 100 % | 0 - 1 % |

Success means walking for the full 20 s on flat ground, or getting all the way up / down the 8-step staircase. The average speed tracking error is 0.04 - 0.07 m/s. You can also continue training from it: `python main.py train --resume pretrained/biped_stairs.pt`.

## How it works

**Robot.** `robot/biped.xml` is a MuJoCo model with a floating torso and two 6-joint legs. The policy outputs 12 joint position targets (offsets from a slightly crouched default pose) at 50 Hz; PD servos inside the simulator turn them into torques at 200 Hz.

**Terrain.** `terrain.py` builds a grid of 10 m tiles: flat ground, "pit" stairs the robot must climb out of, and "pyramid" stairs it must walk down, each at 10 difficulty levels. All robots share one world but each lives in its own copy of the physics state, so they never collide with each other.

**Curriculum.** A stairs robot that reaches the end of its staircase is moved to the next harder level; one that falls is moved down. Robots that beat the hardest level are sent to a random level.

**Observation (105 numbers).** Base linear and angular velocity, gravity direction, the velocity command, joint positions and velocities, the previous action, a gait clock (sin/cos), and a 11x5 height scan of the terrain around the robot.

**Reward.** Track the commanded speed and heading, stay alive, step in rhythm with the gait clock and lift the swing foot clear of the next step, with penalties for wobbling, crouching, sliding feet, jerky actions, torque and joint limits. All weights are in `RewardCfg` in `config.py`.

**PPO.** `agent.py` is a compact PPO: separate actor and critic MLPs, running observation normalisation, GAE, clipped surrogate + clipped value loss, and a KL-adaptive learning rate. Timeouts are bootstrapped so "the episode ran out of time" is not treated like "the robot fell".

## Project layout

```text
sim-robot-trainer/
├── main.py           command line interface
├── config.py         every tunable number (simulation, terrain, rewards, PPO)
├── environment.py    batched MuJoCo environment (observations, rewards, resets, curriculum)
├── terrain.py        stairs generation + fast terrain height lookup
├── agent.py          actor-critic networks and the PPO algorithm
├── train.py          training loop, logging, checkpoints
├── play.py           viewer, video recording, evaluation
├── robot/biped.xml   the robot model (MJCF)
├── pretrained/       a trained policy to try out
├── docs/             images for this README
├── models/           your checkpoints (created by training)
└── logs/             TensorBoard logs (created by training)
```

## Troubleshooting

- **`device: cpu` on a machine with an NVIDIA GPU**: reinstall PyTorch from the CUDA index (see Setup), and update the NVIDIA driver.
- **The viewer window does not open**: update your graphics driver; on Linux a desktop session is required. `python main.py record` works without a window.
- **Robots never get past low stairs levels**: train longer, or look at `episode/reward/*` in TensorBoard to see which term dominates.
- **Changed the observation or the robot?** Old checkpoints will not fit any more; start a fresh run.

## Repository

GitHub: [Usman-Siddiqi/sim-robot-trainer](https://github.com/Usman-Siddiqi/sim-robot-trainer)
