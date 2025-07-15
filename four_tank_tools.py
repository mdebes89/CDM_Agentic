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

def apply_action(input_str: str) -> str:
    """
    Stub for LangChain: we receive exactly the LLM’s JSON
    (e.g. '{"a1":5,"a2":7}'), and just echo it back.
    Your train_manager will then parse and call env.step().
    """
    # sanity‐check it’s valid JSON
    json.loads(input_str)
    return input_str


obs_tool = Tool.from_function(
    func=read_observation,
    name="read_observation",
    description="Returns the current tank heights and setpoints as a list."
)


act_tool = Tool.from_function(
    func=apply_action,
    name="apply_action",
    description=(
        "Accepts a JSON object with exactly two keys, “a1” and “a2” (floats from 0.0 to 10.0), "
        "and returns that same JSON verbatim."
    ),
)
