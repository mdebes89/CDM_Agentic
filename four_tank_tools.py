# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:58:44 2025

@author: mdbs1
"""

from langchain.tools import StructuredTool

def read_observation(obs):
    # expects obs = [h1,h2,h3,h4,h3_SP,h4_SP]
    return {
        "h":     obs[:4],
        "h3_SP": obs[6],
        "h4_SP": obs[7],
    }

# No-op tool to capture the JSON action output
def set_action(action_input: list[float]) -> list[float]:
    # Defensive: coerce length 2 and float clamp
    if not isinstance(action_input, list) or len(action_input) != 2:
        raise ValueError("action_input must be list of length 2")
    a1 = float(action_input[0])
    a2 = float(action_input[1])
    # clamp to [0,10]
    a1 = max(0.0, min(10.0, a1))
    a2 = max(0.0, min(10.0, a2))
    return [a1, a2]

set_action_tool = StructuredTool.from_function(
    func=set_action,
    name="set_action",
    description="Return the chosen two pump settings [a1,a2] (floats in [0,10]).",
    args_schema={
        "type": "object",
        "properties": {
            "action_input": {
                "type": "array",
                "items": {"type":"number"},
                "minItems": 2,
                "maxItems": 2
            }
        },
        "required": ["action_input"]
    }
) 


obs_tool = StructuredTool.from_function(
    func=read_observation,
    name="read_observation",
    description="..."
)

set_action_spec = {
    "name": "set_action",
    "description": "Return the chosen two pump settings [a1,a2] (floats in [0,10]).",
    "parameters": {
        "type": "object",
        "properties": {
            "action_input": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2
            }
        },
        "required": ["action_input"]
    }
} 
