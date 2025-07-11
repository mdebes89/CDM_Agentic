# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:10:53 2025

@author: mdbs1

Holds all the “knobs” for your agentic setup:

agentic and agentic_manager flags toggle between deterministic vs. learned executors and whether the manager can use conditional roles.

ROLE_COSTS is a dict mapping each role name (e.g. "validator_x1", "aggregator") to a scalar cost that the manager pays whenever it engages that role.
"""

ROLE_COSTS = {
    # Four-tank roles must match your executor.py names:
    "validator_h3": 0.01,
    "validator_h4": 0.01,
    "actionizer_h3": 0.05,
    "actionizer_h4": 0.05,
    "conditional_role": 0.02,
    "aggregate_actions": 0.02,
}
