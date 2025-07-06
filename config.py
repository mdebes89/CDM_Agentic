# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:10:53 2025

@author: mdbs1

Holds all the “knobs” for your agentic setup:

agentic and agentic_manager flags toggle between deterministic vs. learned executors and whether the manager can use conditional roles.

ROLE_COSTS is a dict mapping each role name (e.g. "validator_x1", "aggregator") to a scalar cost that the manager pays whenever it engages that role.
"""

ROLE_COSTS = {
    # CSTR cost - Not applied in this implementation
    "validator_T":   0.01,
    "actionizer_T":  0.05,
    "validator_C":   0.01,
    "actionizer_C":  0.05,

    # Four‐tank roles
    "validator_x1":   0, # 0.01,
    "actionizer_x1":  0, # 0.05,
    "validator_x2":   0, # 0.01,
    "actionizer_x2":  0, # 0.05,

}

agentic = False # config flag between deterministic or agentic
agentic_manager = False