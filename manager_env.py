# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:58 2025

@author: mdbs1
"""

import numpy as np
import gymnasium as gym
from cstr_env import make_cstr_env
from config import agentic

# Conditional imports for executor logic
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
        # underlying PC-Gym CSTR env (with a_space of shape (2,))
        self.env = make_cstr_env()
        obs, _ = self.env.reset()

        # expose correct spaces to SB3
        # obs: already set by CSTR env + FlattenObservation wrapper
        self.observation_space = self.env.observation_space

        # manager chooses 4 binary flags
        self.action_space = gym.spaces.MultiBinary(4)

        # define order of control vars matching a_space
        self.control_vars = ["coolant_flow", "feed_rate"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info

    def step(self, manager_action):
        obs = self.current_obs
        flags = manager_action  # length-4 array of 0/1

        # 1 run validators & actionizers based on flags
        proposed = []
        if flags[0] and validator_T(obs):
            proposed.append(actionizer_T(obs))
        if flags[1] and validator_C(obs):
            proposed.append(actionizer_C(obs))
        if flags[2]:
            # note: conditional takes list of dicts
            proposed = [conditional_role(aggregate_actions(proposed))]
        if flags[3]:
            final_dict = aggregate_actions(proposed)
        else:
            final_dict = proposed[0] if proposed else {}

        # 2️ convert dict → 2-element array [coolant_flow, feed_rate]
        action = np.array(
            [ final_dict.get(var, 0.0) for var in self.control_vars ],
            dtype=np.float32
        )

        # 3️ step through PC-Gym
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_obs = next_obs
        return next_obs, reward, terminated, truncated, info