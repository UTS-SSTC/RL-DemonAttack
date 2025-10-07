import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, ResizeObservation, RecordEpisodeStatistics, FrameStackObservation


class RewardConfig:
    def __init__(
            self,
            mode="clip",  # "clip" | "raw" | "scaled"
            clip_low=-1.0, clip_high=1.0,
            scale_divisor=100.0,
            use_life_penalty=False,  # only if terminal_on_life_loss=False
            life_penalty=-1.0,
            use_step_cost=False,
            step_cost=-0.01,
            idle_window=30,  # steps without positive reward before step_cost
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
    Transform ALE raw reward (Δscore) to shaped reward for training.
    Keep raw_reward in info for logging/evaluation comparability.
    """

    def __init__(self, env, cfg: RewardConfig):
        super().__init__(env)
        self.cfg = cfg
        self._frames_since_hit = 0
        self._last_positive_step = -10 ** 9
        self._t = 0
        self._prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames_since_hit = 0
        self._last_positive_step = -10 ** 9
        self._t = 0
        self._prev_lives = info.get("lives", self._prev_lives)
        return obs, info

    def step(self, action):
        obs, raw_r, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(raw_r)

        # Base shaping
        if self.cfg.mode == "clip":
            if raw_r > 0:
                shaped = 1.0
            elif raw_r < 0:
                shaped = -1.0
            else:
                shaped = 0.0
        elif self.cfg.mode == "scaled":
            shaped = float(raw_r) / float(self.cfg.scale_divisor)
        else:
            shaped = float(raw_r)

        # Helpers (optional ablations)
        if raw_r > 0:
            # streak bonus if another hit occurs within streak_window frames
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


def make_env(render_mode=None, stack=4, screen_size=84, terminal_on_life_loss=True, reward_cfg: RewardConfig = None):
    gym.register_envs(ale_py)
    base = gym.make('ALE/DemonAttack-v5', render_mode=render_mode,
                    frameskip=1,
                    repeat_action_probability=0.0)
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
