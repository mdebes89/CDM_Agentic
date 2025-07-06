# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:18 2025

@author: mdbs1
Same interface as executor.py, but with hand-coded (deterministic) validator and actionizer functions:

Used as a baseline and control group or for initial debugging when agentic=False.

"""
import numpy as np

def validator_x1(obs):
    T = obs[1]
    return T > 330.0  

def actionizer_x1(obs: np.ndarray) -> dict:
    T = obs[1]
    delta = min(0.1 * (T - 350.0), 1.0)
    return {"u1": -delta}

def validator_x2(obs):
    Ca, Ca_SP = obs[0], obs[2]
    # if concentration drops >5% below SP
    return Ca < 0.95 * Ca_SP

def actionizer_x2(obs: np.ndarray) -> dict:
    C = obs[0]
    delta = min(0.1 * (0.5 - C), 1.0)
    return {"u2": +delta}

def conditional_role(actions):
    if "u1" in actions and "u2" in actions:
        if actions["u1"] < 0 and actions["u2"] > 0:
            # arbitrary resolution
            return {"u1": actions["u1"] * 0.5}
    return {}

def aggregate_actions(actions_list):
    agg = {}
    for a in actions_list:
        for k, v in a.items():
            agg[k] = agg.get(k, 0) + v
    return agg