# Sim Robot Trainer

This project trains a simple biped robot in PyBullet using PPO from Stable Baselines3. It was built as a reinforcement learning practice project, with the main goal of understanding how environment design, rewards, and training setup affect locomotion.

## What the project does

- builds a custom robot in simulation
- trains a policy with PPO
- saves trained models to `models/`
- lets you list, select, and run saved models from the command line
- logs training data for TensorBoard

## Requirements

- Python 3.8+
- Git

Main Python packages are listed in [requirements.txt](C:\Users\usman\OneDrive - Queen's University\2025 Fall\APSC 143\Practise Code\sim-robot-trainer\requirements.txt), including PyBullet, Stable Baselines3, PyTorch, Gymnasium, and TensorBoard.

## Setup

Clone the repository and install dependencies:

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

Install packages:

```bash
pip install -r requirements.txt
```

Check the CLI:

```bash
python main.py --help
```

## Usage

Train a model with default settings:

```bash
python main.py --train
```

Train for a custom number of timesteps:

```bash
python main.py --train --timesteps 2000000
```

Run a short training pass for testing:

```bash
python main.py --quick_test
```

List saved models:

```bash
python main.py --list
```

Run a saved model:

```bash
python main.py --run --model_path models/your_model.zip --episodes 10
```

Open the interactive model selector:

```bash
python main.py --interactive
```

Render during training if needed:

```bash
python main.py --train --render_training
```

Disable rendering when running a trained model:

```bash
python main.py --run --model_path models/your_model.zip --no_render
```

## Command line options

Available actions:

- `--train`
- `--run`
- `--list`
- `--interactive`
- `--quick_test`

Training options:

- `--timesteps`
- `--render_training`

Run options:

- `--model_path`
- `--episodes`
- `--no_render`

General option:

- `--models_dir`

## Project layout

```text
sim-robot-trainer/
├── agent.py
├── config.py
├── environment.py
├── main.py
├── train.py
├── requirements.txt
├── models/
└── logs/
```

## Notes on the implementation

- `main.py` handles the CLI and dispatches to training or evaluation.
- `environment.py` defines the robot environment and simulation logic.
- `train.py` contains the training loop and quick test entry point.
- `agent.py` creates and loads PPO agents.
- `config.py` stores the main hyperparameters and file paths.

Some of the values in [config.py](C:\Users\usman\OneDrive - Queen's University\2025 Fall\APSC 143\Practise Code\sim-robot-trainer\config.py) are tuned for experimentation rather than realism, so training results depend heavily on reward scaling and episode settings.

## Monitoring

TensorBoard logs are written under `logs/`. To inspect training:

```bash
tensorboard --logdir logs
```

## Troubleshooting

If `pybullet` or another package is missing, make sure the virtual environment is active and reinstall dependencies:

```bash
pip install -r requirements.txt
```

If no saved models appear, check that training completed and that `.zip` files were written to `models/`.

If training feels too slow, avoid rendering and use `--quick_test` while debugging.

## Repository

GitHub: [Usman-Siddiqi/sim-robot-trainer](https://github.com/Usman-Siddiqi/sim-robot-trainer)
