# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:18 2025

@author: mdbs1
"""

def validator_T(obs):
    T = obs["T"]
    return T > 350.0  # threshold

def actionizer_T(obs):
    T = obs["T"]
    delta = min(0.1 * (T - 350), 1.0)
    return {"coolant_flow": -delta}

def validator_C(obs):
    C = obs["C_A"]
    return C < 0.5

def actionizer_C(obs):
    C = obs["C_A"]
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