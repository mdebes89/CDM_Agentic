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
        # underlying CSTR
        self.env = make_cstr_env()

        # get a sample raw obs to build flat space
        raw_obs, _ = self.env.reset()
        flat_obs = self._flatten(raw_obs)

        # set spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )
        self.action_space = gym.spaces.MultiBinary(4)

        # ordering for actions
        self.control_vars = ["coolant_flow", "feed_rate"]

        # storage
        self.current_raw_obs = None

    def _flatten(self, raw_obs: dict) -> np.ndarray:
        # same order as o_space: [Ca, T, Ca_SP]
        return np.array([
            raw_obs["Ca"],
            raw_obs["T"],
            raw_obs["Ca_SP"]
        ], dtype=np.float32)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self.current_raw_obs = raw_obs
        return self._flatten(raw_obs), info

    def step(self, manager_action):
        # use raw obs for validators/actionizers
        raw = self.current_raw_obs
        flags = manager_action  # array of 4 zeros/ones

        # run reasoning roles
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

        # dict → array for env action
        action = np.array(
            [ final_dict.get(var, 0.0) for var in self.control_vars ],
            dtype=np.float32
        )

        # step underlying CSTR
        next_raw, reward, terminated, truncated, info = self.env.step(action)
        self.current_raw_obs = next_raw

        # flatten for SB3
        next_flat = self._flatten(next_raw)
        return next_flat, reward, terminated, truncated, info