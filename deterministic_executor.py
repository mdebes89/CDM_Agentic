# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:18 2025

@author: mdbs1
Same interface as executor.py, but with hand-coded (deterministic) validator and actionizer functions:

Used as a baseline and control group or for initial debugging when agentic=False.

"""
import numpy as np

def validator_x1(obs):
    """
    Trigger pump-1 agent if tank 3 deviates >5% from its setpoint.
    obs = [h1,h2,h3,h4,h1_SP,h2_SP,h3_SP,h4_SP]
    """
    h3, h3_SP = obs[2], obs[6]    # SP for h3 lives at obs[6]
    return abs(h3 - h3_SP) > 0.05 * h3_SP

def actionizer_x1(obs: np.ndarray) -> dict:
    """
    Simple P-controller for pump 1:
      u1 = clip(1.0 * (h3_SP - h3), 0.0, 1.0)
    """
    h3, h3_SP = obs[2], obs[6]
    # full voltage command ∈ [0,1]
    u1 = np.clip(1.0 * (h3_SP - h3), 0.0, 1.0)
    return {"u1": u1}

def validator_x2(obs):
    """
    Trigger pump-2 agent if tank 4 deviates >5% from its setpoint.
    """
    h4, h4_SP = obs[3], obs[7]    # SP for h4 lives at obs[7]
    return abs(h4 - h4_SP) > 0.05 * h4_SP

def actionizer_x2(obs: np.ndarray) -> dict:
    """
    Simple P-controller for pump 2:
      u2 = clip(1.0 * (h4_SP - h4), 0.0, 1.0)
    """
    h4, h4_SP = obs[3], obs[7]
    # full voltage command ∈ [0,1]
    u2 = np.clip(1.0 * (h4_SP - h4), 0.0, 1.0)
    return {"u2": u2}

def conditional_role(actions):
    # if both pumps want to act in opposing directions, reduce one
    if "u1" in actions and "u2" in actions:
        if actions["u1"] < 0 and actions["u2"] > 0:
            return {"u1": actions["u1"] * 0.5}
    return {}

def aggregate_actions(actions_list):
    # simple sum of all delta‐voltage proposals
    agg = {}
    for a in actions_list:
        for k, v in a.items():
            agg[k] = agg.get(k, 0) + v
    return agg