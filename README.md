# Sim Robot Trainer

Teaches a simulated two-legged robot to walk and climb stairs with reinforcement learning (PPO).

Thousands of robots are simulated in parallel with [MuJoCo](https://mujoco.org/), a
terrain curriculum moves them from flat ground to taller and taller stairs, and a
from-scratch PPO implementation in PyTorch learns the controller.

Built as a reinforcement learning practice project, with the main goal of understanding
how the robot model, environment design, rewards and training setup affect locomotion.

## What it does

- **12-DoF biped** (`robot/biped.xml`): hip yaw / roll / pitch, knee, ankle pitch / roll per leg, ~25 kg, PD position servos with torque limits
- **Tasks**: follow a commanded speed and heading on flat ground, climb stairs up, walk stairs down (step heights 3 cm → 16 cm over 10 curriculum levels)
- **Fast**: physics for all robots is stepped by MuJoCo's multi-threaded rollout on every CPU core, the neural networks run on the GPU
- **Tools**: interactive viewer, video export, evaluation table, TensorBoard logging, resumable checkpoints

Tuned for a Ryzen 7 8845HS + RTX 4050 laptop, but it runs on anything (a GPU is optional).

## Setup

Python 3.10+.

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
python main.py play
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

What to expect: flat-ground walking appears after roughly 100-150 iterations (~5-7 million steps), the stairs curriculum needs on the order of 1500-2000 iterations. `python main.py bench` tells you how long an iteration takes on your machine.

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

The console prints one line every 10 iterations:

```text
it   400/2000 | 120,000 steps/s (sim 0.3s, learn 0.1s) | return 45.86 | ep len 1000.0 | fell 0.00 | stairs lvl 3.2 ok 0.81 | std 0.28 | lr 2e-04 | eta 12.0 min
```

`stairs lvl` is the average curriculum level of the stairs robots (0 = 3 cm steps, 9 = 16 cm steps) and `ok` is the fraction of stairs episodes that reached the end of the staircase.

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
