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
        from four_tank_env import make_four_tank_env, o_space, SP
        # sample new initial h1–h4 each reset
        h_low, h_high = o_space["low"][:4], o_space["high"][:4]
        initial_h     = np.random.uniform(h_low, h_high).astype(np.float32)
        initial_sp    = np.array([
            SP["h1"][0], SP["h2"][0], SP["h3"][0], SP["h4"][0]
            ], dtype=np.float32)
        x0 = np.concatenate([initial_h, initial_sp])

        # pass x0 into your four‐tank factory
        self.env = make_four_tank_env(x0=x0)
        # Reset underlying env, store raw
        obs, info = self.env.reset(**kwargs)    # obs is ndarray
        #print("raw obs range:", obs[:4].min(), obs[:4].max())
        self.current_raw_obs = obs
        self.current_step = 0
        return obs, info

    def step(self, manager_action):
        # 1) Undo any [-1,+1] normalization so raw is in engineering units
        obs_norm = self.current_raw_obs
        low  = self.observation_space.low
        high = self.observation_space.high
        # if x_norm = 2*(x - low)/(high-low) - 1, then
        # x = ((x_norm + 1)/2)*(high - low) + low
        raw = ((obs_norm + 1.0) * 0.5) * (high - low) + low
        #print(f"[SANITY] unnormalized h1–h4 = {raw[:4]}")
        
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

        # conditional wrapper - Only necessary if Agentic conditional wrapper is needed
        if agentic_manager and flags[2]:
            engaged_roles.append("conditional")
            proposed = [conditional_role(aggregate_actions(proposed))]
            

        # Always aggregate all proposals by default (assumed management function)
        final_dict = aggregate_actions(proposed)
           
        if self.debugging:    
            print(f"[SANITY] proposals = {proposed}")
            print(f"[SANITY] merged final_dict = {final_dict}")

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
        
        # 6) Compute your new, shaped tracking reward in [0,1]
        # raw = your un-normalized state vector from earlier in this function
        h3, sp3 = raw[2], raw[4]
        h4, sp4 = raw[3], raw[5]
        norm_err3 = abs(h3 - sp3) / sp3
        norm_err4 = abs(h4 - sp4) / sp4
        shaped    = 1.0 - 0.5 * (norm_err3 + norm_err4)  # ∈ [0,1]
    
        # 7) (Optional) subtract any compute cost or add bonus
        cost            = sum(ROLE_COSTS[r] for r in engaged_roles)
        engagement_bonus = 0.1 * len(engaged_roles)
    
        # 8) Final manager reward
        manager_reward = shaped  # or: shaped - cost + engagement_bonus # By design we exlcude cost and engagement to reduce complexity
        
        if self.debugging:
           print(f"[DEBUG] perf_reward={perf_reward:.3f}, cost={cost:.3f}, manager_reward={manager_reward:.3f}")

        # 7) Advance control clock
        self.current_step += 1
        
        # 8) Log for debugging/monitoring
        info["perf_reward"]   = perf_reward
        info["manager_cost"]  = cost
        info["manager_reward"] = manager_reward
        
        
        return next_obs, manager_reward, terminated, truncated, info