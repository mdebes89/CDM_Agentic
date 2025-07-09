# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:18 2025

@author: mdbs1
Same interface as executor.py, but with hand-coded (deterministic) validator and actionizer functions:

Used as a baseline and control group or for initial debugging when agentic=False.
obs = [h1, h2, h3, h4, h1_SP, h2_SP, h3_SP, h4_SP]

"""
import numpy as np
# import inspect

Kp = 10.0 # P controller. Max range of pump inputs
EPSILON = 0.01       # threshold on squared‐error (e.g. 0.1 m deviation²)

def validator_x1(obs):
    """
    Trigger pump-1 agent when the squared error (h3–SP3)^2 > EPSILON.
    """
    h3, sp3 = obs[2], obs[6]
    return (h3 - sp3)**2 > EPSILON

def actionizer_x1(obs: np.ndarray) -> dict:
    """
    Simple P-controller for pump 1:
      u1 = clip(Kp * (h3_SP - h3), 0.0, Kp)
    """   
    # drive tank 3 → h3 against obs[4] (the env’s h3_SP)
    h3, sp3 = obs[2], obs[6]
    error = sp3 - h3
    raw_u = Kp* error
    #print(f"[SANITY][x1] h3={h3:.3f}, sp3={sp3:.3f}, error={error:.3f}, raw_u={raw_u:.3f}")
    u1 = np.clip(raw_u, 0.0, Kp)
    return {"u1": np.float32(u1)}

def validator_x2(obs):
    """
    Trigger pump-2 agent when the squared error (h4–SP4)^2 > EPSILON.
    """
    h4, sp4 = obs[3], obs[7]
    return (h4 - sp4)**2 > EPSILON

def actionizer_x2(obs: np.ndarray) -> dict:
    """
    Simple P-controller for pump 2:
      u2 = clip(1.0 * (h4_SP - h4), 0.0, 1.0)
    """
    # drive tank 4 → h4 against obs[5] (the env’s h4_SP)
    h4, sp4 = obs[3], obs[7]
    error = sp4 - h4
    raw_u = Kp * error
    # print(f"[DEBUG-x2] raw obs = {obs}  (len={len(obs)})")
    u2 = np.clip(raw_u, 0.0, Kp)
    return {"u2": np.float32(u2)}

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