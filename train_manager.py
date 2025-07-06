# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:08:57 2025

@author: mdbs1
"""

from manager_env import HierarchicalManagerEnv
from stable_baselines3 import PPO

def main():
    env = HierarchicalManagerEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("ppo_manager_cstr")
    
    obs, _ = env.reset()
    for _ in range(50):
        action, _ = model.predict(obs)
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()