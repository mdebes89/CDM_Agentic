# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:58 2025

@author: mdbs1
"""

import gymnasium as gym
from cstr_env import make_cstr_env
from config import agentic

# Conditional imports
if agentic:
    from executor import (
        validator_T, validator_C,
        actionizer_T, actionizer_C,
        conditional_role, aggregate_actions
    )
else:
    from deterministic_executor import (
        validator_T, validator_C,
        actionizer_T, actionizer_C,
        conditional_role, aggregate_actions
    )


class HierarchicalManagerEnv(gym.Env):
    def __init__(self):
        self.env = make_cstr_env()
        obs, _ = self.env.reset()
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.MultiBinary(4) # 4 binary flags for role execution
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info
    
    def step(self, manager_action):
        obs = self.current_obs
        flags = manager_action
    
        proposed = []
        if flags[0] and validator_T(obs):
            proposed.append(actionizer_T(obs))
        if flags[1] and validator_C(obs):
            proposed.append(actionizer_C(obs))
        if flags[2]:
            proposed.append(conditional_role(aggregate_actions(proposed)))
        if flags[3]:
            final_action = aggregate_actions(proposed)
        else:
            final_action = proposed[0] if proposed else {}
    
        next_obs, reward, terminated, truncated, info = self.env.step(final_action)
        self.current_obs = next_obs
        return next_obs, reward, terminated, truncated, info