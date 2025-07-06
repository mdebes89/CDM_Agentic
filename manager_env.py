# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:58 2025

@author: mdbs1
"""

import numpy as np
import gymnasium as gym
from first_order_env import make_first_order_env
from config import agentic, ROLE_COSTS

# Conditional imports for executor logic
if agentic:
    from executor import (
        validator_x1, validator_x2,
        actionizer_x1, actionizer_x2,
        conditional_role, aggregate_actions
    )
else:
    from deterministic_executor import (
        validator_x1, validator_x2,
        actionizer_x1, actionizer_x2,
        conditional_role, aggregate_actions
    )


class HierarchicalManagerEnv(gym.Env):
    def __init__(self):
        self.env = make_first_order_env()
        # directly reuse the FO env’s spaces:
        self.observation_space = self.env.observation_space  # Box(shape=(3,),…)
        self.action_space      = gym.spaces.MultiBinary(4)   # manager picks 4 flag
        # 4) Define control variable order for a_space of shape (2,)
        self.control_vars = ["u1", "u2"]

        # 5) Storage for raw obs
        self.current_raw_obs = None

    def reset(self, **kwargs):
        # Reset underlying env, store raw
        obs, info = self.env.reset(**kwargs)    # obs is ndarray
        self.current_raw_obs = obs
        return obs, info

    def step(self, manager_action):
        # 1) Use  dict for logic
        raw = self.current_raw_obs
        flags = manager_action  # 4-length 0/1 array
        
        # 2) Track which roles actually ran
        engaged_roles = []

        # 3) Executors produce a dict of control changes
        proposed = []
        # validator_x1
        if flags[0]:
            engaged_roles.append("validator_x1")
            if validator_x1(raw):
                engaged_roles.append("actionizer_x1")
                proposed.append(actionizer_x1(raw))

        # validator_x2
        if flags[1]:
            engaged_roles.append("validator_x2")
            if validator_x2(raw):
                engaged_roles.append("actionizer_x2")
                proposed.append(actionizer_x2(raw))

        # conditional wrapper
        if flags[2]:
            engaged_roles.append("conditional")
            proposed = [conditional_role(aggregate_actions(proposed))]

        # aggregator
        if flags[3]:
            engaged_roles.append("aggregator")
            final_dict = aggregate_actions(proposed)
        else:
            final_dict = proposed[0] if proposed else {}

        # 4) Flatten into the two-element [u1, u2] array
        action = np.array([
            final_dict.get("u1", 0.0),
            final_dict.get("u2", 0.0),
        ], dtype=np.float32)


        # 5) Step the CSTR
        next_obs, perf_reward, terminated, truncated, info = self.env.step(action)
        self.current_raw_obs = next_obs
        
        # 6) Compute and subtract the cost
        cost = sum(ROLE_COSTS[r] for r in engaged_roles)
        manager_reward = perf_reward - cost

        
        # 7) Log for debugging/monitoring
        info["perf_reward"]   = perf_reward
        info["manager_cost"]  = cost
        info["manager_reward"] = manager_reward
        
        return next_obs, manager_reward, terminated, truncated, info