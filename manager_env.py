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
        self.env = make_cstr_env()
        # directly reuse the CSTR env’s spaces:
        self.observation_space = self.env.observation_space  # Box(shape=(3,),…)
        self.action_space      = self.env.action_space       # Box(shape=(2,), low=[-1,-1], high=[1,1])
        # 4) Define control variable order for a_space of shape (2,)
        self.control_vars = ["coolant_flow", "feed_rate"]

        # 5) Storage for raw obs
        self.current_raw_obs = None

    def _flatten(self, raw_obs: dict) -> np.ndarray:
        # Map the PC-Gym dict to a fixed vector [Ca, T, Ca_SP]
        return np.array([
            raw_obs["Ca"],
            raw_obs["T"],
            raw_obs["Ca_SP"]
        ], dtype=np.float32)

    def reset(self, **kwargs):
        # Reset underlying env, store raw
        obs, info = self.env.reset(**kwargs)    # obs is ndarray
        self.current_raw_obs = obs
        return obs, info

    def step(self, manager_action):
        # 1) Use raw dict for logic
        raw = self.current_raw_obs
        flags = manager_action  # 4-length 0/1 array

        # 2) Executors produce a dict of control changes
        proposed = []
        if flags[0] and validator_T(raw):
            proposed.append(actionizer_T(raw))
        if flags[1] and validator_C(raw):
            proposed.append(actionizer_C(raw))
        if flags[2]:
            proposed = [conditional_role(aggregate_actions(proposed))]
        if flags[3]:
            final_dict = aggregate_actions(proposed)
        else:
            final_dict = proposed[0] if proposed else {}

        # 3) Flatten that dict → a 2-element array
        action = np.array(
            [ final_dict.get(var, 0.0) for var in self.control_vars ],
            dtype=np.float32
        )

        # 4) Step the CSTR
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.current_raw_obs = next_obs
        return next_obs, reward, terminated, truncated, info