# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:58:44 2025

@author: mdbs1
"""

from langchain_core.tools import Tool

def read_observation(obs):
    # expects obs = [h1,h2,h3,h4,h3_SP,h4_SP]
    return {
        "h":     obs[:4],
        "h3_SP": obs[4],
        "h4_SP": obs[5],
    }

def apply_action(input_str: str) -> str:
    """
    Expects input_str like '5.2,3.7'.
    Parses and returns the same normalized '5.2,3.7' string.
    The manager will split and env.step() it.
    """
    a1_str, a2_str = [s.strip() for s in input_str.split(",")]
    # validate ranges
    a1, a2 = float(a1_str), float(a2_str)
    if not (0 <= a1 <= 10 and 0 <= a2 <= 10):
        raise ValueError(f"Valve settings out of range: {a1}, {a2}")
    # echo back
    return f"{a1},{a2}"
 


obs_tool = Tool.from_function(
    func=read_observation,
    name="read_observation",
    description="Returns the current tank heights and setpoints as a list."
)


act_tool = Tool.from_function(
    func=apply_action,
    name="apply_action",
    description="Take two floats in [0,10] separated by a comma, e.g. '4.2,5.8', and echo them.",
)