# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:11:28 2025

@author: mdbs1
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# 1. Validator Role Template
validator_template = ChatPromptTemplate.from_template(
    """
You are a process control validator.
Given the current state:
{obs}

Determine whether the following condition is true:
"{condition_description}"

Respond with "yes" or "no".
"""
)

def call_validator(obs, condition_description):
    prompt = validator_template.format_messages(obs=str(obs), condition_description=condition_description)
    response = llm(prompt)
    return "yes" in response.content.lower()

def validator_T(obs):
    return call_validator(obs, "Temperature T > 350 K")

def validator_C(obs):
    return call_validator(obs, "Concentration of A (C_A) < 0.5 mol/L")

# 2. Actionizer Role Template
action_template = ChatPromptTemplate.from_template(
    """
You are a chemical process controller.
Given the current observation:
{obs}

Condition triggered: {condition_label}

Recommend an action by specifying:
- control_variable: one of ["coolant_flow", "feed_rate"]
- adjustment: positive or negative float value

Respond in JSON:
{{"control_variable": "...", "adjustment": float}}
"""
)

action_output_schema = [
    ResponseSchema(name="control_variable", description="Name of control variable to adjust"),
    ResponseSchema(name="adjustment", description="Numeric adjustment to apply")
]
parser = StructuredOutputParser.from_response_schemas(action_output_schema)

def call_actionizer(obs, condition_label):
    prompt = action_template.format_messages(obs=str(obs), condition_label=condition_label)
    response = llm(prompt)
    return parser.parse(response.content)

def actionizer_T(obs):
    return call_actionizer(obs, "High temperature (T > 350 K)")

def actionizer_C(obs):
    return call_actionizer(obs, "Low concentration (C_A < 0.5 mol/L)")

# 3. Conditional Role: Combine and Check Constraints
conditional_template = ChatPromptTemplate.from_template(
    """
You are a conflict resolver in a chemical plant.
Given the proposed actions:
{actions}

Decide if there are conflicts between them (e.g., coolant and feed_rate affecting each other).
Respond with:
- adjustments: updated list of recommended changes in JSON form:
[{{"control_variable": "...", "adjustment": float}}, ...]
"""
)

def conditional_role(actions):
    prompt = conditional_template.format_messages(actions=str(actions))
    response = llm(prompt)
    return parser.parse(response.content)

# 4. Aggregator Role: Finalize Action
aggregator_template = ChatPromptTemplate.from_template(
    """
You are an aggregator agent.
Given the following candidate control actions:
{candidates}

Decide the best single action to apply next. Output format:
{{"control_variable": "...", "adjustment": float}}
"""
)

def aggregate_actions(candidates):
    prompt = aggregator_template.format_messages(candidates=str(candidates))
    response = llm(prompt)
    return parser.parse(response.content)