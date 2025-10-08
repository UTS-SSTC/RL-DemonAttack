# Deep Q-Network for Atari DemonAttack

PyTorch implementation of Deep Q-Networks (DQN) and its variants for mastering the Atari game DemonAttack.

---

## 📋 Problem Description

### DemonAttack Environment

**DemonAttack** is a classic Atari 2600 game where the player defends against waves of demons on the ice planet Krybor. The objective is to survive enemy attacks and maximize score by destroying demons.

**Environment Specifications:**
- **Action Space**: `Discrete(6)` - NOOP, FIRE, LEFT, RIGHT, LEFTFIRE, RIGHTFIRE
- **Observation Space**: `Box(0, 255, (210, 160, 3), uint8)` RGB frames
- **Challenge**: Requires precise timing, spatial awareness, and strategic decision-making

**Game Mechanics:**
- Player starts with 3 reserve bunkers (can increase up to 6)
- Surviving each wave without damage grants an additional bunker
- Enemy hit destroys one bunker; losing all bunkers ends the game
- Score increases by destroying different types of demons

### Research Challenge

The goal is to train a deep reinforcement learning agent that can:
1. Learn effective policies from raw pixel inputs
2. Achieve human-level or superhuman performance
3. Generalize across different enemy patterns and game states

**Baseline Benchmarks:**
- Random agent: ~50 points
- Human average: ~1,971 points
- Human expert: ~3,401 points
- Published DQN: 3,000-9,000 points
- Rainbow DQN: 100,000+ points

---

## 🧠 Approach & Model Architecture

This project implements multiple DQN variants with increasing sophistication:

### 1. Baseline DQN (Mnih et al., Nature 2015)

**Architecture:**
```
Input: [batch, 4, 84, 84] (4 stacked grayscale frames)
  ↓
Conv2d(4→32, kernel=8, stride=4) + ReLU
  ↓
Conv2d(32→64, kernel=4, stride=2) + ReLU
  ↓
Conv2d(64→64, kernel=3, stride=1) + ReLU
  ↓
Flatten → [batch, 3136]
  ↓
Linear(3136→512) + ReLU
  ↓
Linear(512→n_actions) → Q-values
```

**Key Techniques:**
- **Experience Replay**: Store and sample past transitions to break temporal correlation
- **Target Network**: Stabilize training with slowly updated Q-value targets
- **ε-greedy Exploration**: Balance exploration and exploitation
- **Frame Stacking**: Use 4 consecutive frames to capture motion information

### 2. Dueling DQN (Wang et al., ICML 2016)

Separates Q-value computation into two streams:

```
Shared Features (Conv layers)
  ↓
  ├─→ Value Stream: V(s) ─────────┐
  │                                ↓
  └─→ Advantage Stream: A(s,a) ───→ Q(s,a) = V(s) + [A(s,a) - mean(A)]
```

**Advantages:**
- Better learning of state values independent of actions
- More stable training for environments with many redundant actions
- Improved performance on games requiring precise action selection

### 3. Noisy Dueling DQN (Fortunato et al., 2017)

Replaces ε-greedy with learned parametric noise in network weights:

**NoisyLinear Layer:**
```
W = μ_w + σ_w ⊙ ε_w  (training)
W = μ_w              (evaluation)
```

Where:
- `μ_w`: Learnable mean weights
- `σ_w`: Learnable noise scale
- `ε_w`: Factorized Gaussian noise

**Benefits:**
- Automatic exploration without manual ε scheduling
- Better late-stage exploration for complex strategies
- State-dependent exploration (different noise per state)

### Additional Improvements

**Double DQN** (van Hasselt et al., 2015):
- Reduces Q-value overestimation
- Uses online network for action selection, target network for evaluation

**Prioritized Experience Replay** (Schaul et al., 2015):
- Samples important transitions more frequently
- Weighted importance sampling for unbiased gradient updates

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
uv run python -m dqn_demon_attack.scripts.train_yaml --config configs/baseline_dqn.yaml

# Dueling DQN (2M steps, ~8-12 hours, recommended)
uv run python -m dqn_demon_attack.scripts.train_yaml --config configs/dueling_dqn.yaml

# Noisy Dueling DQN (3M steps, ~15-20 hours, best performance)
uv run python -m dqn_demon_attack.scripts.train_yaml --config configs/noisy_dueling_dqn.yaml
```

### Evaluation

```bash
# Evaluate trained model
python -m dqn_demon_attack.scripts.eval --ckpt runs/dueling_dqn/checkpoints/final.pt --episodes 10

# Watch agent play
python -m dqn_demon_attack.scripts.watch --ckpt runs/dueling_dqn/checkpoints/final.pt --mode human
```

### Visualization

```bash
# Plot training curves
python -m dqn_demon_attack.utils.viz --log runs/dueling_dqn/train_log.csv
```

---

## ⚙️ Configuration

### Available Configurations

| Config | Model | Steps | Training Time | Target Score |
|--------|-------|-------|---------------|--------------|
| `baseline_dqn.yaml` | DQN | 500k | 2-4h | ~150 |
| `dueling_dqn.yaml` ⭐ | DuelingDQN | 2M | 8-12h | ~400-600 |
| `noisy_dueling_dqn.yaml` 🏆 | NoisyDuelingDQN | 3M | 15-20h | ~800-1000+ |

### Configuration Template

```yaml
# Experiment Settings
exp_name: my_experiment
total_steps: 2000000
eval_every: 100000
seed: 42
device: cuda  # or 'cpu'

# Model Settings
model_type: DuelingDQN  # DQN | DuelingDQN | NoisyDQN | NoisyDuelingDQN
reward_mode: clip       # clip | scaled | raw

# Training Hyperparameters
gamma: 0.99             # Discount factor
lr: 0.00025             # Learning rate
batch_size: 64          # Batch size
replay_size: 200000     # Replay buffer capacity
warmup: 20000           # Random warmup steps
target_update_freq: 2000  # Target network update frequency
grad_clip: 10.0         # Gradient clipping threshold

# Exploration (ignored for Noisy Networks)
eps_start: 1.0
eps_end: 0.01
eps_decay_steps: 1000000

# Advanced Features
use_double_dqn: true
use_prioritized_replay: false
```

### Key Hyperparameters

**Learning Rate (`lr`)**:
- Baseline: 1e-4
- Recommended: 2.5e-4
- Higher values risk instability

**Replay Buffer (`replay_size`)**:
- Minimum: 50k
- Recommended: 200k
- Maximum: 500k (for best models)

**Batch Size (`batch_size`)**:
- Small (32): Faster updates, higher variance
- Large (128): More stable, slower updates
- Recommended: 64

**Target Update Frequency (`target_update_freq`)**:
- Too frequent (<1k): Training instability
- Too rare (>10k): Slow adaptation
- Recommended: 2k-5k steps

---

## 📊 Expected Results

### Performance Comparison

| Method | Score | Training Steps | Time (GPU) | Improvement |
|--------|-------|----------------|------------|-------------|
| Random | ~50 | - | - | Baseline |
| Baseline DQN | ~150 | 500k | 2-4h | 3x |
| Dueling DQN | ~400-600 | 2M | 8-12h | 8-12x |
| Noisy Dueling DQN | ~800-1000+ | 3M | 15-20h | 16-20x |
| Human Average | ~1,971 | - | - | 39x |
| Human Expert | ~3,401 | - | - | 68x |

### Training Dynamics

**Baseline DQN**:
- Initial phase (0-100k): Random exploration, score ~40-60
- Learning phase (100k-300k): Policy improvement, score ~80-120
- Convergence (300k-500k): Stable performance, score ~130-150
- High variance (±90 points)

**Dueling DQN**:
- Warmup (0-100k): Initial exploration, score ~50-80
- Rapid learning (100k-500k): Major improvements, score ~150-300
- Refinement (500k-2M): Strategy optimization, score ~400-600
- Medium variance (±120 points)

**Noisy Dueling DQN**:
- Early exploration (0-200k): Noisy network learning, score ~60-100
- Stable learning (200k-1M): Consistent improvements, score ~200-500
- Advanced strategies (1M-3M): Complex behaviors, score ~800-1000+
- Lower variance (±150 points)

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
│   └── scripts/
│       ├── train_yaml.py        # Training script
│       ├── eval.py              # Evaluation script
│       └── watch.py             # Visualization script
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

---

## 🔬 Implementation Details

### Preprocessing Pipeline

1. **Grayscale Conversion**: RGB → grayscale (210×160×3 → 210×160)
2. **Resizing**: 210×160 → 84×84 (bilinear interpolation)
3. **Frame Stacking**: Stack 4 consecutive frames → 4×84×84
4. **Normalization**: [0, 255] → [0.0, 1.0]
5. **Reward Clipping**: Clip rewards to [-1, 0, +1]

### Training Algorithm

```python
1. Initialize replay buffer D, Q-network Q(θ), target network Q̂(θ⁻)
2. For each episode:
    a. Reset environment, get initial state s₀
    b. For each timestep:
        - Select action: a = argmax Q(s,a) with prob. 1-ε, else random
        - Execute action a, observe reward r, next state s'
        - Store transition (s, a, r, s', done) in D
        - Sample minibatch from D
        - Compute target: y = r + γ·max Q̂(s',a')
        - Update Q-network: minimize (Q(s,a) - y)²
        - Periodically update target network: θ⁻ ← θ
```

### Loss Function

**Baseline/Dueling DQN**:
```
L(θ) = 𝔼[(r + γ·max Q̂(s',a') - Q(s,a))²]
```

**Double DQN**:
```
L(θ) = 𝔼[(r + γ·Q̂(s', argmax Q(s',a')) - Q(s,a))²]
```

**Prioritized Replay**:
```
L(θ) = 𝔼[w_i · (r + γ·Q̂(s',a') - Q(s,a))²]
where w_i = (N · P(i))⁻ᵝ  (importance weight)
```

---

## 🛠️ Troubleshooting

### Out of Memory
- Reduce `batch_size` to 32
- Reduce `replay_size` to 100k
- Close other GPU applications

### Low Performance
- Train longer (`total_steps` → 2M+)
- Use better architecture (`DuelingDQN` or `NoisyDuelingDQN`)
- Increase replay buffer (`replay_size` → 200k+)
- Adjust learning rate (`lr` → 2.5e-4)

### Training Instability
- Reduce learning rate (`lr` → 1e-4)
- Increase batch size (`batch_size` → 128)
- Enable gradient clipping (`grad_clip: 10.0`)
- Increase target update frequency (`target_update_freq` → 5k)

---

## 📚 References

### Core Papers

1. **DQN**: Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
2. **Double DQN**: van Hasselt et al. (2015). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.
3. **Dueling DQN**: Wang et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." *ICML*.
4. **Prioritized Replay**: Schaul et al. (2015). "Prioritized Experience Replay." *ICLR*.
5. **Noisy Networks**: Fortunato et al. (2017). "Noisy Networks for Exploration." *ICLR*.
6. **Rainbow**: Hessel et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning." *AAAI*.

### Implementation

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - Atari game environments
- [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) - Arcade Learning Environment

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

This implementation is based on foundational work by DeepMind and the reinforcement learning research community. Special thanks to the authors of the papers listed in the references section.
