# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:58:44 2025

@author: mdbs1
"""

from langchain_core.tools import Tool
import json

def read_observation(obs):
    # expects obs = [h1,h2,h3,h4,h3_SP,h4_SP]
    return {
        "h":     obs[:4],
        "h3_SP": obs[4],
        "h4_SP": obs[5],
    }

def apply_action(a1: float, a2: float) -> str:
    """
    LLM “chooses” valve settings; returns them as JSON.
    Your manager loop will parse out a1, a2 and call env.step().
    """
    return json.dumps({"a1": a1, "a2": a2})


obs_tool = Tool.from_function(
    func=read_observation,
    name="read_observation",
    description="Returns the current tank heights and setpoints as a list."
)


raw_act_tool = Tool.from_function(
    func=apply_action,
    name="apply_action",
    description="Choose two valve settings (floats 0.0–10.0)."
)
