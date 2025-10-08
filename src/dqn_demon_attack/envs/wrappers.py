"""
Environment wrappers and utilities for DemonAttack-v5.

This module provides custom reward shaping and environment creation
for the Atari DemonAttack game using Gymnasium.
"""

import ale_py
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    ResizeObservation,
    RecordEpisodeStatistics,
    FrameStackObservation
)


class RewardConfig:
    """
    Configuration for reward shaping strategies.

    Supports multiple reward modes: 'clip', 'scaled', or 'raw'.
    Additional features include life penalties, step costs, and streak bonuses.

    Args:
        mode: Reward transformation mode. Options are 'clip', 'scaled', or 'raw'.
        clip_low: Lower bound for clipping reward (used in 'clip' mode).
        clip_high: Upper bound for clipping reward (used in 'clip' mode).
        scale_divisor: Divisor for scaling rewards (used in 'scaled' mode).
        use_life_penalty: Whether to apply penalty when agent loses a life.
        life_penalty: Penalty value when losing a life.
        use_step_cost: Whether to apply cost for idle steps.
        step_cost: Cost per idle step.
        idle_window: Number of steps without positive reward before applying step cost.
        use_streak_bonus: Whether to apply bonus for consecutive hits.
        streak_window: Time window (in steps) for streak bonus.
        streak_bonus: Bonus value for consecutive hits within streak window.
    """

    def __init__(
        self,
        mode="clip",
        clip_low=-1.0,
        clip_high=1.0,
        scale_divisor=100.0,
        use_life_penalty=False,
        life_penalty=-1.0,
        use_step_cost=False,
        step_cost=-0.01,
        idle_window=30,
        use_streak_bonus=False,
        streak_window=12,
        streak_bonus=0.1,
    ):
        self.mode = mode
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.scale_divisor = scale_divisor
        self.use_life_penalty = use_life_penalty
        self.life_penalty = life_penalty
        self.use_step_cost = use_step_cost
        self.step_cost = step_cost
        self.idle_window = idle_window
        self.use_streak_bonus = use_streak_bonus
        self.streak_window = streak_window
        self.streak_bonus = streak_bonus


class DemonAttackReward(gym.Wrapper):
    """
    Custom reward wrapper for DemonAttack environment.

    Transforms raw ALE rewards (delta score) into shaped rewards suitable
    for training. The original raw reward is preserved in the info dict
    for logging and evaluation purposes.

    Args:
        env: The base Gymnasium environment.
        cfg: RewardConfig object specifying reward shaping parameters.
    """

    def __init__(self, env, cfg: RewardConfig):
        super().__init__(env)
        self.cfg = cfg
        self._frames_since_hit = 0
        self._last_positive_step = -10 ** 9
        self._t = 0
        self._prev_lives = None

    def reset(self, **kwargs):
        """Reset the environment and internal tracking variables."""
        obs, info = self.env.reset(**kwargs)
        self._frames_since_hit = 0
        self._last_positive_step = -10 ** 9
        self._t = 0
        self._prev_lives = info.get("lives", self._prev_lives)
        return obs, info

    def step(self, action):
        """
        Execute action and return shaped reward.

        Args:
            action: Action to execute in the environment.

        Returns:
            Tuple of (observation, shaped_reward, terminated, truncated, info).
            The info dict contains 'raw_reward' with the original reward value.
        """
        obs, raw_r, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(raw_r)

        raw_r_float = float(raw_r)

        if self.cfg.mode == "clip":
            if raw_r_float > 0:
                shaped = 1.0
            elif raw_r_float < 0:
                shaped = -1.0
            else:
                shaped = 0.0
        elif self.cfg.mode == "scaled":
            shaped = raw_r_float / float(self.cfg.scale_divisor)
        else:
            shaped = raw_r_float

        if raw_r_float > 0:
            if self.cfg.use_streak_bonus and (self._t - self._last_positive_step) <= self.cfg.streak_window:
                shaped += self.cfg.streak_bonus
            self._last_positive_step = self._t
            self._frames_since_hit = 0
        else:
            self._frames_since_hit += 1

        if self.cfg.use_step_cost and self._frames_since_hit >= self.cfg.idle_window:
            shaped += self.cfg.step_cost

        if self.cfg.use_life_penalty and "lives" in info:
            lives = info["lives"]
            if self._prev_lives is not None and lives < self._prev_lives:
                shaped += self.cfg.life_penalty
            self._prev_lives = lives

        self._t += 1
        return obs, shaped, terminated, truncated, info


def make_env(
    render_mode=None,
    stack=4,
    screen_size=84,
    terminal_on_life_loss=True,
    reward_cfg: RewardConfig | None = None
):
    """
    Create a fully wrapped DemonAttack environment.

    Applies standard Atari preprocessing including grayscale conversion,
    frame skipping, resizing, and frame stacking. Adds custom reward shaping.

    Args:
        render_mode: Rendering mode for the environment ('human', 'rgb_array', or None).
        stack: Number of frames to stack.
        screen_size: Size of the resized observation (height and width).
        terminal_on_life_loss: Whether to end episode when agent loses a life.
        reward_cfg: RewardConfig object. If None, uses default clip mode.

    Returns:
        Wrapped Gymnasium environment ready for training or evaluation.
    """
    gym.register_envs(ale_py)
    base = gym.make(
        'ALE/DemonAttack-v5',
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0
    )
    env = RecordEpisodeStatistics(base)
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        frame_skip=4,
        scale_obs=False,
        terminal_on_life_loss=terminal_on_life_loss,
        screen_size=screen_size,
    )
    env = ResizeObservation(env, (screen_size, screen_size))
    env = FrameStackObservation(env, stack)

    if reward_cfg is None:
        reward_cfg = RewardConfig(mode="clip")
    env = DemonAttackReward(env, reward_cfg)

    return env
