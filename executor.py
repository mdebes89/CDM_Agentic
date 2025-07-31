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
3. Conditional role to resolve conflicting proposals. Not implemented.
4. Aggregator role to merge candidate adjustments into final actions. Not implemented.

This script is not fully implemented / used during our training script.

"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import re

# Instantiate a deterministic LLM for control decisions
llm = ChatOpenAI(model="gpt-4", temperature=0.0)

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

**Reference:** full environment spec here → https://maximilianb2.github.io/pc-gym/env/four_tank/
'''  
)

def call_validator(obs, index, deadband=0.03):
    prompt = validator_template.format_messages(
        obs=str(obs), tank=f"h{index}", index=str(index), deadband=deadband
    )
    response = llm(prompt)
    resp = response.content.strip().lower()
    return bool(re.match(r"^yes$", resp))

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
- Valid adjustment range: {valve_min:.1f} to {valve_max:.1f} (float).

Respond in JSON: {{"control_variable": "<u1|u2>", "adjustment": <float>}}

– Use a positive adjustment to increase flow, negative to decrease.
– Don’t output any other keys or commentary.

**Reference:** full environment spec here → https://maximilianb2.github.io/pc-gym/env/four_tank/
'''  
)

def call_actionizer(obs, index):
    # Compute error
    heights = obs[:4]
    setpoints = obs[4:]
    err = heights[index-1] - setpoints[index-3]
    VALVE_MIN, VALVE_MAX = 0.0, 10.0 # Env limit. Could be dervied from env directly.
    # Build the initial message list
    messages = action_template.format_messages(
        obs=str(obs),
        tank=f"h{index}",
        index=str(index),
        error=err,
        valve_min=VALVE_MIN,   # exact match to {valve_min}
        valve_max=VALVE_MAX,   # exact match to {valve_max}
    )
    # ← Here we append the parser’s own instructions so the model outputs valid JSON
    format_instructions = parser.get_format_instructions()
    messages.append({"text": format_instructions})

    response = llm(messages)
    # Parse with the StructuredOutputParser
    return parser.parse(response.content)

# Two actionizers: for h3->u1 and h4->u2

def actionizer_h3(obs):
    return call_actionizer(obs, index=3)

def actionizer_h4(obs):
    return call_actionizer(obs, index=4)
