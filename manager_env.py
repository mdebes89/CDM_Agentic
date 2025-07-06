# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:58 2025

@author: mdbs1

Wraps the four-tank plant with a hierarchical decision layer (the “manager”):

On each step it takes a 4-bit manager_action flag vector → decides which of your child roles to invoke (validators, actionizers, aggregator, etc.).

It collects their proposed actuator commands into a final_dict, flattens that to the 2-D action [u1,u2], steps the plant, then computes

It tracks debugging info so you can see which roles ran, what they proposed, and the resulting cost/performanc
"""

import numpy as np
import gymnasium as gym
from four_tank_env  import make_four_tank_env 
from config import agentic, agentic_manager, ROLE_COSTS

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
    def __init__(self, debugging=False):
        self.debugging = debugging
        self.env = make_four_tank_env()
        # after disabling normalise_o, grab the raw (un-scaled) space
        # observation_space_base is the Box([low…], [high…]) you defined as o_space
        self.observation_space = self.env.observation_space_base  
        # manager now picks 3 flags: validator_x1, validator_x2, conditional
        self.action_space      = gym.spaces.MultiBinary(3)
        # 4) Define control variable order for a_space of shape (2,)
        self.control_vars = ["u1", "u2"]

        # 5) Storage for raw obs
        self.current_raw_obs = None
        self.current_step = 0

    def reset(self, **kwargs):
        # Reset underlying env, store raw
        obs, info = self.env.reset(**kwargs)    # obs is ndarray
        self.current_raw_obs = obs
        self.current_step = 0
        return obs, info

    def step(self, manager_action):
        # 1) Use  dict for logic
        raw = self.current_raw_obs
        flags = [bool(manager_action[i]) for i in range(3)]
        
        
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
        if agentic_manager and flags[2]:
            engaged_roles.append("conditional")
            proposed = [conditional_role(aggregate_actions(proposed))]
            
        # if nobody proposed anything, fall back to a simple P‐controller on tank1
        if not proposed:
            sp1 = self.env.SP["h1"][self.current_step]
            Kp  = 1.0  # choose an appropriate proportional gain
            proposed.append({"u1": Kp*(sp1 - raw[0]),
                             "u2": 0.0})

        # Always aggregate all proposals by default (assumed management function)
        final_dict = aggregate_actions(proposed)
           
        if self.debugging:    
            print("   proposed raw outputs:", proposed)
            print("   final_dict before flatten:", final_dict)

        # 4) Flatten into the two-element [u1, u2] array
        action = np.array([
            final_dict.get("u1", 0.0),
            final_dict.get("u2", 0.0),
        ], dtype=np.float32)

        if self.debugging:
            print(f"[DEBUG] step={self.current_step}, engaged_roles={engaged_roles}, action={action}")
        # 5) Step the four-tank plant
        next_obs, perf_reward, terminated, truncated, info = self.env.step(action)
        if self.debugging:
            print(f"[DEBUG] next_obs[:4]={next_obs[:4]}, sp={self.env.SP['h1'][self.current_step]}")
        self.current_raw_obs = next_obs
        
        # 6) Compute and subtract the cost
        cost = sum(ROLE_COSTS[r] for r in engaged_roles)
        engagement_bonus = 0.1 * len(engaged_roles) # Bonus reward for engaging roles
        manager_reward = perf_reward# - cost + engagement_bonus
        
        if self.debugging:
           print(f"[DEBUG] perf_reward={perf_reward:.3f}, cost={cost:.3f}, manager_reward={manager_reward:.3f}")

        # 7) Advance control clock
        self.current_step += 1
        
        # 8) Log for debugging/monitoring
        info["perf_reward"]   = perf_reward
        info["manager_cost"]  = cost
        info["manager_reward"] = manager_reward
        
        return next_obs, manager_reward, terminated, truncated, info