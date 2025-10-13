# Sim-Robot-Trainer

A reinforcement learning project focused on training a simulated bipedal robot to walk using PyBullet physics simulation and Stable Baselines3 PPO algorithm.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyBullet](https://img.shields.io/badge/PyBullet-3.2.5+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🤖 Demo

*[A GIF or video of the trained robot walking will be added here once training is complete.]*

## 📖 Table of Contents

- [Project Goal](#-project-goal)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Installation](#️-installation)
- [Usage](#️-usage)
  - [Training a New Agent](#training-a-new-agent)
  - [Running a Pre-trained Agent](#running-a-pre-trained-agent)
  - [Available Commands](#available-commands)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Robot Design](#-robot-design)
- [Reward System](#-reward-system)
- [Monitoring Training](#-monitoring-training)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Future Work](#-future-work)
- [License](#-license)

## 🎯 Project Goal

The primary objective of this project is to explore reinforcement learning fundamentals by training a bipedal robot to develop a stable walking gait in a physics-based simulation. Starting from random actions, the agent learns through trial and error using the Proximal Policy Optimization (PPO) algorithm.

**Key Learning Objectives:**
- Understanding agent-environment interaction in RL
- Implementing reward shaping for locomotion tasks
- Working with continuous action spaces
- Physics simulation integration with ML frameworks

## ✨ Features

- **Custom Bipedal Robot**: Programmatically created using PyBullet primitives
- **PPO Algorithm**: State-of-the-art policy gradient method for continuous control
- **Comprehensive CLI**: Interactive training and evaluation interface
- **Real-time Monitoring**: TensorBoard integration for training visualization
- **Modular Architecture**: Clean, maintainable code structure
- **Robust Reward System**: Multi-component reward function for stable walking
- **Checkpoint Management**: Automatic model saving and loading

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|------------|----------|
| **Language** | Python | 3.8+ |
| **Physics Simulation** | PyBullet | 3.2.5+ |
| **RL Framework** | Stable Baselines3 | 2.0.0+ |
| **Deep Learning** | PyTorch | 1.13.0+ |
| **Environment Interface** | Gymnasium | 0.28.0+ |
| **Monitoring** | TensorBoard | 2.10.0+ |
| **Numerical Computing** | NumPy | 1.21.0+ |
| **Visualization** | Matplotlib | 3.5.0+ |

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Usman-Siddiqi/sim-robot-trainer.git
   cd sim-robot-trainer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python main.py --help
   ```

## ▶️ Usage

### Training a New Agent

Start training with default parameters:
```bash
python main.py --train
```

Customize training parameters:
```bash
python main.py --train --timesteps 2000000 --render_training
```

Quick test training (shorter session):
```bash
python main.py --quick_test
```

### Running a Pre-trained Agent

Run a specific model:
```bash
python main.py --run --model_path models/robot_walker_final_20231012_143000.zip --episodes 10
```

Interactive model selection:
```bash
python main.py --interactive
```

### Available Commands

| Command | Description |
|---------|-------------|
| `--train` | Start training a new model |
| `--run` | Run a pre-trained model |
| `--list` | List all available trained models |
| `--interactive` | Interactive model selection and running |
| `--quick_test` | Run a short training session for testing |

**Training Options:**
- `--timesteps INT`: Number of training timesteps (default: 1,000,000)
- `--render_training`: Enable visual rendering during training

**Running Options:**
- `--model_path PATH`: Path to the trained model file
- `--episodes INT`: Number of episodes to run (default: 5)
- `--no_render`: Disable visual rendering

## 📂 Project Structure

```
sim-robot-trainer/
├── 📁 models/              # Saved model files and checkpoints
├── 📁 logs/                # Training logs and TensorBoard data
├── 📄 main.py              # CLI entry point and model execution
├── 📄 train.py             # Training loop and callbacks
├── 📄 agent.py             # PPO agent configuration and utilities
├── 📄 environment.py       # Robot environment and physics setup
├── 📄 config.py            # Hyperparameters and configuration
├── 📄 requirements.txt     # Project dependencies
├── 📄 .gitignore          # Git ignore rules
└── 📄 README.md           # This file
```

## ⚙️ Configuration

Key hyperparameters can be modified in `config.py`:

```python
# Training Parameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
MAX_EPISODE_STEPS = 2000

# Environment Parameters
SIMULATION_TIMESTEP = 1/240
ROBOT_START_HEIGHT = 1.0
FALL_THRESHOLD = 0.3

# Reward Parameters
FORWARD_REWARD_SCALE = 100.0
FALL_PENALTY = -100.0
```

## 🤖 Robot Design

The bipedal robot is programmatically created with:
- **Torso**: Main body (box shape)
- **4 Controllable Joints**: 2 hips + 2 knees
- **6 Links Total**: Upper legs, lower legs, and feet
- **Physics Properties**: Realistic mass distribution and damping

**Action Space**: 4 continuous values (joint torques)  
**Observation Space**: 17 values including joint states, torso orientation, and velocities

## 🎯 Reward System

The reward function encourages walking behavior through multiple components:

- ✅ **Forward Progress**: Rewards movement in positive X direction
- ⚖️ **Stability**: Penalizes excessive tilting (roll/pitch)
- 🏃 **Upright Stance**: Bonus for maintaining minimum height
- ❌ **Fall Penalty**: Large negative reward for falling

## 📊 Monitoring Training

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard/
```

Training metrics include:
- Episode rewards
- Policy loss
- Value function loss
- Entropy (exploration measure)

## 🔧 Troubleshooting

### Common Issues

**"No module named 'pybullet'"**
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt`

**Training is too slow**
- Disable rendering: Remove `--render_training` flag
- Reduce timesteps for testing: Use `--quick_test`

**Model files not found**
- Check if `models/` directory exists
- Use `--list` to see available models

**GPU not detected**
- The project uses CPU by default for stability
- GPU support can be enabled by modifying device settings in `agent.py`

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 🚀 Future Work

Planned enhancements:

- [ ] **Advanced Robot Models**: More complex bipedal designs with additional joints
- [ ] **Terrain Variety**: Training on slopes, stairs, and uneven surfaces
- [ ] **Obstacle Navigation**: Dynamic obstacle avoidance during walking
- [ ] **Gait Analysis**: Detailed biomechanical analysis tools
- [ ] **Multi-Agent Training**: Training multiple robots simultaneously
- [ ] **Real Robot Transfer**: Sim-to-real transfer learning capabilities
- [ ] **Advanced Algorithms**: Implementation of SAC, TD3, and other RL algorithms
- [ ] **Web Interface**: Browser-based training monitoring and control

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author**: Usman Siddiqi  
**Contact**: [GitHub Profile](https://github.com/Usman-Siddiqi)  
**Project Link**: [sim-robot-trainer](https://github.com/Usman-Siddiqi/sim-robot-trainer)

⭐ **Star this repository if you find it helpful!**