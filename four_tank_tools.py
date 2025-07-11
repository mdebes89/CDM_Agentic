# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:58:44 2025

@author: mdbs1
"""

from langchain.tools import Tool

def read_observation(env):
    obs = env.get_observation()  # length 8: [h1,h2,h3,h4,h1_SP,h2_SP,h3_SP,h4_SP]
    # we only need the two lower-tank heights + their setpoints
    sub = [obs[i] for i in (0,1,2,3,6,7)]
    return {"h": sub}

def apply_action(env, a1: float, a2: float):
    """
    Apply valve voltages directly in [0,10].
    """
    env.step([a1, a2])
    return {"status": "applied"}

obs_tool = Tool.from_function(
    func=read_observation,
    name="read_observation",
    description="Returns the current tank heights and setpoints as a list."
)

act_tool = Tool.from_function(
    func=apply_action,
    name="apply_action",
    description="Apply two valve settings between 0 and 10 to the four-tank environment."
)