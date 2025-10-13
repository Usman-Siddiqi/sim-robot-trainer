# Sim-Robot-Trainer

A reinforcement learning project focused on training a simulated bipedal robot to walk using PyBullet physics simulation and Stable Baselines3 PPO algorithm.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyBullet](https://img.shields.io/badge/PyBullet-3.2.5+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-orange.svg)

> **⚠️ Development Status**: This project is currently under active development. Recent major improvements to robot physics and reward systems have been implemented. Training results and documentation are being updated.

## 🤖 Demo

*[A GIF or video of the trained robot walking will be added here once current training cycle is complete.]*

## 📖 Table of Contents

- [Project Goal](#-project-goal)
- [Features](#-features)
- [Development Status](#-development-status)
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

## 🚧 Development Status

**Current Phase**: Physics Optimization & Retraining

**Recent Major Updates (October 2025):**
- ✅ **Fixed Critical Robot Physics Bug**: Resolved phantom collision geometry that was affecting training
- ✅ **Improved Mass Distribution**: Updated to realistic 28kg total mass with proper weight distribution
- ✅ **Optimized Reward Function**: Balanced multi-component rewards for natural walking behavior
- ✅ **Enhanced Joint Control**: Implemented joint-specific torque limits (120 N·m hips, 60 N·m knees)
- ✅ **Corrected Robot Geometry**: Fixed starting height and fall thresholds for realistic physics

**Next Steps:**
- 🔄 **Active Training**: Currently retraining with corrected physics (5M timesteps)
- 📊 **Performance Analysis**: Evaluating improved walking behavior and convergence
- 📹 **Documentation Update**: Recording new walking demonstrations
- 🎯 **Hyperparameter Tuning**: Fine-tuning reward components based on results

**Known Issues:**
- Previous models trained before physics fixes are deprecated
- Training results from pre-October 2025 should be ignored
- Demo videos/GIFs pending completion of current training cycle

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

> **Note**: Pre-trained models from before October 2025 are deprecated due to physics fixes.

Run a specific model:
```bash
python main.py --run --model_path models/robot_walker_final_20251013_143000.zip --episodes 10
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
- `--timesteps INT`: Number of training timesteps (default: 5,000,000)
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
# Training Parameters (Updated)
LEARNING_RATE = 3e-4
GAMMA = 0.99
MAX_EPISODE_STEPS = 1500  # ~6 seconds at 240Hz
TOTAL_TIMESTEPS = 5000000  # Optimized for quality

# Environment Parameters (Corrected Physics)
SIMULATION_TIMESTEP = 1/240
ROBOT_START_HEIGHT = 0.83  # Realistic standing height
FALL_THRESHOLD = 0.4       # 50% of standing height

# Reward Parameters (Balanced)
FORWARD_REWARD_SCALE = 10.0    # Reduced for balance
FALL_PENALTY = -10.0           # Reduced for exploration
STABILITY_PENALTY_SCALE = 2.0  # New stability component
UPRIGHT_REWARD = 0.5           # Upright bonus

# Joint Limits (Realistic)
HIP_MAX_TORQUE = 120.0    # Hips need more torque
KNEE_MAX_TORQUE = 60.0    # Knees need less
```

## 🤖 Robot Design

The bipedal robot features a corrected physics model:
- **Torso**: Main body (9.0kg, box shape)
- **4 Controllable Joints**: 2 hips + 2 knees with realistic torque limits
- **6 Links Total**: Upper legs (4.0kg each), lower legs (4.0kg each), feet (1.5kg each)
- **Total Mass**: 28kg (realistic for 0.8m leg span)
- **Physics Properties**: Proper mass distribution, realistic inertia, and corrected joint chains

**Action Space**: 4 continuous values (joint torques, properly clipped)  
**Observation Space**: 17 values including joint states, torso orientation, and velocities

## 🎯 Reward System

The reward function encourages natural walking through balanced components:

- ✅ **Forward Progress**: Rewards movement in positive X direction (10x scale)
- ⚖️ **Stability**: Penalizes excessive tilting with roll/pitch penalties (2x scale)
- 🏃 **Upright Stance**: Small bonus for maintaining minimum height (0.5x scale)
- ❌ **Fall Penalty**: Moderate negative reward for falling (-10.0)
- 🎛️ **Action Smoothness**: Prevents jerky movements (0.01x scale)

**Reward Balance**: Designed for ~1.4:1 forward:stability ratio for natural gaits

## 📊 Monitoring Training

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard/
```

Key metrics to watch:
- **Episode Reward**: Should increase over training
- **Episode Length**: Should grow as robot learns to walk
- **Forward Distance**: Total distance covered per episode
- **Policy Loss**: Should decrease and stabilize
- **Value Function Loss**: Should converge

## 🔧 Troubleshooting

### Common Issues

**"No module named 'pybullet'"**
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt`

**Training is too slow**
- Disable rendering: Remove `--render_training` flag
- Reduce timesteps for testing: Use `--quick_test`
- Check CPU temperature (should be <85°C for sustained training)

**Model files not found**
- Check if `models/` directory exists
- Use `--list` to see available models
- Note: Pre-October 2025 models are deprecated

**Robot appears broken/has red artifacts**
- This was fixed in October 2025 - update to latest code
- Delete old models and retrain from scratch

**GPU not detected**
- The project uses CPU by default for stability
- GPU support can be enabled by modifying device settings in `agent.py`

### Development Notes

**For Contributors:**
- All models trained before October 13, 2025 should be considered invalid
- The robot physics underwent major corrections - expect different behavior
- Training now requires ~4-6 hours for 5M timesteps on modern CPUs
- Check `debug_robot_structure()` if implementing robot modifications

## 🤝 Contributing

Contributions are welcome! Current areas of interest:
- **Training Optimization**: Hyperparameter tuning for faster convergence
- **Advanced Rewards**: More sophisticated reward shaping
- **Robot Variations**: Different robot morphologies
- **Environment Enhancements**: Terrain variety, obstacles
- **Documentation**: Training results, performance analysis

Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 🚀 Future Work

Planned enhancements:

**Short Term (Q4 2025):**
- [ ] **Complete Current Training**: Finish 5M timestep training cycle
- [ ] **Performance Benchmarking**: Detailed analysis of walking metrics
- [ ] **Demo Videos**: Record and publish walking demonstrations
- [ ] **Hyperparameter Study**: Systematic reward component analysis

**Medium Term (2026):**
- [ ] **Advanced Robot Models**: More complex bipedal designs with additional joints
- [ ] **Terrain Variety**: Training on slopes, stairs, and uneven surfaces
- [ ] **Obstacle Navigation**: Dynamic obstacle avoidance during walking
- [ ] **Gait Analysis**: Detailed biomechanical analysis tools

**Long Term:**
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

---

*Last Updated: October 13, 2025 - Major physics corrections and retraining in progress*