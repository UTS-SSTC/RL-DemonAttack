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

# Dueling DQN (2M steps, ~8-12 hours, recommended & life-loss terminal)
uv run train --config configs/dueling_dqn.yaml

# Dueling DQN (3M steps, scaled rewards + full episodes)
uv run train --config configs/dueling_dqn_scaled.yaml

# Noisy Dueling DQN (3M steps, best performance)
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
| `dueling_dqn.yaml` ⭐ | DuelingDQN | 2M | 8-12h | ~650 |
| `dueling_dqn_scaled.yaml` | DuelingDQN | 3M | 10-14h | ~650 (higher variance) |
| `noisy_dueling_dqn.yaml` 🏆 | NoisyDuelingDQN | 3M | 15-20h | ~900+ |

Key hyperparameters: RMSprop (`lr=6.25e-5 → 2–2.5e-5` linear decay, `alpha=0.95`, `eps=1e-5`), `batch_size=32`, `replay_size≤180k` (uint8 storage keeps RAM <16 GB), `target_update_freq=4k–8k`, evaluation over ≥20 episodes. Use `reward_mode: scaled` + `terminal_on_life_loss: false` (see `dueling_dqn_scaled.yaml`) when you specifically want longer-horizon rewards; otherwise stick to clipped rewards for stability.

### Reward / Termination Switches

- `reward_mode: clip` reproduces the classic Nature DQN signal and keeps TD errors well behaved.
- `reward_mode: scaled` preserves raw magnitudes, which can boost very long episodes but may destabilize the value scale. Pair it with a smaller learning rate or longer decay.
- `terminal_on_life_loss: true` ends episodes after each life, matching the ALE evaluation protocol and providing denser resets.
- `terminal_on_life_loss: false` runs through all lives for smoother scoring curves but can inflate variance.

Each config in `configs/` sets these knobs explicitly so you can revert/advance with a single flag change.

---

## 📊 Expected Results

| Method | Score | Steps | Time |
|--------|-------|-------|------|
| Random | ~50 | - | - |
| Baseline DQN | ~150 | 500k | 2-4h |
| Dueling DQN | ~400-650 | 3M | 10-14h |
| Noisy Dueling DQN | ~900-1100 | 4M | 18-24h |
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

Initial rsults
```
# Baseline DQN
[Eval] raw return 97.8±82.8 | len 394
        mean Q 2.908 | Q std 0.069
        best/avg/worst episode = 420.0/97.8/0.0
        score per frame ≈ 0.2482
        
# Dueling DQN
[Eval] raw return 320.3±485.5 | len 752
        mean Q 6.773 | Q std 0.154
        best/avg/worst episode = 3415.0/320.3/30.0
        score per frame ≈ 0.4261

# Noisy Dueling DQN
[Eval] raw return 836.5±840.0 | len 1164
        mean Q 5.511 | Q std 0.196
        best/avg/worst episode = 3655.0/836.5/20.0
        score per frame ≈ 0.7189
```