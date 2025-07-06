# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:18 2025

@author: mdbs1
"""
import numpy as np

def validator_T(obs):
    T, Ca_SP = obs[1], obs[2]
    # if T rises 10 K above nominal SP‐temperature (derive SP from model)
    return T > 330.0  

def actionizer_T(obs: np.ndarray) -> dict:
    T = obs[1]
    delta = min(0.1 * (T - 350.0), 1.0)
    return {"coolant_flow": -delta}

def validator_C(obs):
    Ca, Ca_SP = obs[0], obs[2]
    # if concentration drops >5% below SP
    return Ca < 0.95 * Ca_SP

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