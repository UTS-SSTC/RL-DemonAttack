import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/DemonAttack-v5')
print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")