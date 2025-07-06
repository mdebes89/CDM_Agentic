# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:10:53 2025

@author: mdbs1
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

    # Optional: cost for your wrapper roles
    "conditional":    0, # 0.02,
    "aggregator":     0, # 0.03,
}

agentic = False # config flag between deterministic or agentic