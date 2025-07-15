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
        "h3_SP": obs[4],
        "h4_SP": obs[5],
    }

# No-op tool to capture the JSON action output
def set_action(action_input: list[float]) -> list[float]:
    return action_input
 


obs_tool = StructuredTool.from_function(
    func=read_observation,
    name="read_observation",
    description="..."
)


set_action_tool =  StructuredTool.from_function(
    func=set_action,
    name="set_action",
    description="Capture the manager’s [a1,a2]",
    args_schema={
      "type":"object",
      "properties": {
        "action_input": {
          "type":"array","items":{"type":"number"},
          "minItems":2,"maxItems":2
        }
      },
      "required":["action_input"]
    }
)    
