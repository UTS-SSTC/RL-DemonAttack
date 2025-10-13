# Deep Q-Network for Atari DemonAttack

PyTorch implementation of Deep Q-Networks (DQN) and its variants for mastering the Atari game DemonAttack.

---

## 📋 Overview

**DemonAttack** is a classic Atari 2600 game where the player defends against waves of demons. This project trains RL agents to play the game from raw pixel inputs.

**Benchmarks:**
- Random: ~50 | Human avg: ~1,971 | Human expert: ~3,401
- DQN: ~150 | Dueling DQN: ~600 | Noisy Dueling DQN: ~1,000+

**Implemented Models:**
- Baseline DQN (Mnih et al., 2015)
- Dueling DQN (Wang et al., 2016)
- Noisy Dueling DQN (Fortunato et al., 2017)
- Optional: Double DQN, Prioritized Experience Replay

---

## 🚀 Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Training

```bash
# Baseline DQN (500k steps, ~2-4 hours)
uv run train --config configs/baseline_dqn.yaml

# Dueling DQN (2M steps, ~8-12 hours, recommended)
uv run train --config configs/dueling_dqn.yaml

# Noisy Dueling DQN (3M steps, ~15-20 hours, best performance)
uv run train --config configs/noisy_dueling_dqn.yaml
```

### Evaluation

```bash
# Evaluate trained model
uv run eval --ckpt runs/dueling_dqn/checkpoints/final.pt --episodes 10

# Watch agent play
uv run watch --ckpt runs/dueling_dqn/checkpoints/final.pt --mode human
```

### Visualization

```bash
# Plot training curves
uv run viz --log runs/dueling_dqn/train_log.csv
```

### Web GUI

```bash
uv run web  # Open http://localhost:5000
```

Real-time monitoring, video recording, training curves, and model evaluation.

---

## ⚙️ Configuration

| Config | Model | Steps | Time (GPU) | Score |
|--------|-------|-------|------------|-------|
| `baseline_dqn.yaml` | DQN | 500k | 2-4h | ~150 |
| `dueling_dqn.yaml` ⭐ | DuelingDQN | 2M | 8-12h | ~600 |
| `noisy_dueling_dqn.yaml` 🏆 | NoisyDuelingDQN | 3M | 15-20h | ~1000+ |

Key hyperparameters: `lr=2.5e-4`, `batch_size=64`, `replay_size=200k`, `target_update_freq=2k`

---

## 📊 Expected Results

| Method | Score | Steps | Time |
|--------|-------|-------|------|
| Random | ~50 | - | - |
| Baseline DQN | ~150 | 500k | 2-4h |
| Dueling DQN | ~400-600 | 2M | 8-12h |
| Noisy Dueling DQN | ~800-1000+ | 3M | 15-20h |
| Human Average | ~1,971 | - | - |

---

## 📁 Project Structure

```
RL-DemonAttack/
├── configs/                      # Configuration files
│   ├── baseline_dqn.yaml        # Nature DQN baseline
│   ├── dueling_dqn.yaml         # Dueling architecture (recommended)
│   └── noisy_dueling_dqn.yaml   # Best performance
│
├── src/dqn_demon_attack/
│   ├── agents/
│   │   ├── models.py            # DQN architectures
│   │   └── replay.py            # Experience replay buffers
│   ├── envs/
│   │   └── wrappers.py          # Environment preprocessing
│   ├── utils/
│   │   ├── config.py            # Configuration management
│   │   ├── logger.py            # Training logging
│   │   └── viz.py               # Visualization tools
│   ├── scripts/
│   │   ├── train.py             # Training script
│   │   ├── eval.py              # Evaluation script
│   │   └── watch.py             # Visualization script
│   └── web/
│       ├── app.py               # Flask application
│       ├── training_manager.py  # Training session management
│       ├── evaluation_manager.py # Evaluation management
│       └── templates/
│           └── index.html       # Web interface
│
├── runs/                         # Training outputs
│   └── <exp_name>/
│       ├── config.yaml          # Saved configuration
│       ├── train_log.csv        # Training metrics
│       └── checkpoints/         # Model checkpoints
│
├── pyproject.toml               # Dependencies
└── README.md                    # This file
```
