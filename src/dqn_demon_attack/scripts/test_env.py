"""
Quick environment setup test for Atari DemonAttack.

Verifies that the ALE/DemonAttack-v5 environment is properly installed
and displays basic information about the action and observation spaces.
"""

import ale_py
import gymnasium as gym


def main():
    """Test environment creation and display space information."""
    gym.register_envs(ale_py)

    env = gym.make('ALE/DemonAttack-v5')
    print(f"Action space: {env.action_space}")
    print(f"Sample action: {env.action_space.sample()}")

    print(f"Observation space: {env.observation_space}")
    print(f"Sample observation: {env.observation_space.sample()}")


if __name__ == "__main__":
    main()
