# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:11:28 2025

Updated for Four-Tank Environment Control using Agentic Roles

- Observation: [h1, h2, h3, h4, h3_SP, h4_SP]
- Actions: two continuous valve settings in [0,10]
- Deadband: ±5% of setpoint (~0.01–0.03 m)

This file defines:
1. Validator roles that check whether each tank's error is outside the deadband.
2. Actionizer roles that propose control adjustments for each valve.
3. Conditional role to resolve conflicting proposals.
4. Aggregator role to merge candidate adjustments into final actions.
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import math

# Instantiate a deterministic LLM for control decisions\llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# Common parser schema
action_output_schema = [
    ResponseSchema(name="control_variable", description="Valve index: u1 or u2"),
    ResponseSchema(name="adjustment", description="Float adjustment to apply to valve setting")
]
parser = StructuredOutputParser.from_response_schemas(action_output_schema)

# 1. Validator Role Templates
validator_template = ChatPromptTemplate.from_template(
    '''
You are a validator for the Four-Tank system.
Given the current state:
{obs}

Check if the error for tank {tank} i.e. |h{index} - h{index}_SP| exceeds the deadband ({deadband:.3f} m).
Respond with "yes" or "no".
'''  
)

def call_validator(obs, index, deadband=0.03):
    prompt = validator_template.format_messages(
        obs=str(obs), tank=f"h{index}", index=str(index), deadband=deadband
    )
    response = llm(prompt)
    return "yes" in response.content.lower()

# Two validators: for tank 3 and tank 4

def validator_h3(obs):
    return call_validator(obs, index=3)

def validator_h4(obs):
    return call_validator(obs, index=4)

# 2. Actionizer Role Templates
action_template = ChatPromptTemplate.from_template(
    '''
You are an actionizer for the Four-Tank system.
Given the current state:
{obs}

Error for tank {tank}: {error:.3f} m (h{index} - h{index}_SP).
Recommend an adjustment for valve {valve} by specifying:
- control_variable: one of ["u1", "u2"]
- adjustment: a float (positive to increase flow, negative to decrease)

Respond in JSON: {{"control_variable": "<u1|u2>", "adjustment": <float>}}
'''  
)

def call_actionizer(obs, index):
    # Compute error
    heights = obs[:4]
    setpoints = obs[4:]
    err = heights[index-1] - setpoints[index-3]
    valve = f"u{index-2}"  # map tank3->u1, tank4->u2
    prompt = action_template.format_messages(
        obs=str(obs), tank=f"h{index}", index=str(index), error=err, valve=valve
    )
    response = llm(prompt)
    return parser.parse(response.content)

# Two actionizers: for h3->u1 and h4->u2

def actionizer_h3(obs):
    return call_actionizer(obs, index=3)

def actionizer_h4(obs):
    return call_actionizer(obs, index=4)

# 3. Conditional Role: Resolve Conflicts or Combine
conditional_template = ChatPromptTemplate.from_template(
    '''
You are a conflict resolver for the Four-Tank system.
Given the proposed adjustments:
{actions}

If any valve appears in multiple proposals, average their adjustments.
Respond in JSON list of adjustments:
[{"control_variable": "u1", "adjustment": float}, ...]
'''  
)

def conditional_role(actions):
    prompt = conditional_template.format_messages(actions=str(actions))
    response = llm(prompt)
    return parser.parse(response.content)

# 4. Aggregator Role: Final Actions
aggregator_template = ChatPromptTemplate.from_template(
    '''
You are the final aggregator for the Four-Tank control.
Given candidate adjustments:
{candidates}

Select the best single adjustment per valve and output:
{{"u1": <float>, "u2": <float>}}
'''  
)

def aggregate_actions(candidates):
    prompt = aggregator_template.format_messages(candidates=str(candidates))
    response = llm(prompt)
    parsed = parser.parse(response.content)
    # Convert to dict of u1,u2
    out = {parsed["control_variable"]: parsed["adjustment"]}
    return {"u1": out.get("u1", 0.0), "u2": out.get("u2", 0.0)}