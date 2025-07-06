# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:18 2025

@author: mdbs1
"""
import numpy as np

def validator_T(obs: np.ndarray) -> bool:
    # T is the 2nd entry
    return obs[1] > 350.0

def actionizer_T(obs: np.ndarray) -> dict:
    T = obs[1]
    delta = min(0.1 * (T - 350.0), 1.0)
    return {"coolant_flow": -delta}

def validator_C(obs: np.ndarray) -> bool:
    # Ca is the 1st entry
    return obs[0] < 0.5

def actionizer_C(obs: np.ndarray) -> dict:
    C = obs[0]
    delta = min(0.1 * (0.5 - C), 1.0)
    return {"feed_rate": +delta}

def conditional_role(actions):
    if "coolant_flow" in actions and "feed_rate" in actions:
        if actions["coolant_flow"] < 0 and actions["feed_rate"] > 0:
            # arbitrary resolution
            return {"coolant_flow": actions["coolant_flow"] * 0.5}
    return {}

def aggregate_actions(actions_list):
    agg = {}
    for a in actions_list:
        for k, v in a.items():
            agg[k] = agg.get(k, 0) + v
    return agg