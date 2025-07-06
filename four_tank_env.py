# -*- coding: utf-8 -*-
"""
four_tank_env.py

Creates a PC-Gym environment for a four-tank process
with two manipulated variables (pump flows u1, u2).
https://maximilianb2.github.io/pc-gym/#quick-start
"""

import numpy as np
import pcgym

def make_four_tank_env():
    nsteps = 100
    tsim   = 25.0
    # Setpoint trajectories for each tank (constant here)
    # Must match the model’s state names ["h1","h2","h3","h4"]
    SP = {
        "h1": np.ones(nsteps) * 1.0,
        "h2": np.ones(nsteps) * 0.8,
        "h3": np.ones(nsteps) * 0.5,
        "h4": np.ones(nsteps) * 0.3,
    }
    # States = [h1, h2, h3, h4]
    # Setpoints = [SP1, SP2, SP3, SP4]
    # → total obs-vector length = 8
    state_low  = np.zeros(4, dtype=np.float32)
    state_high = np.ones(4, dtype=np.float32)*2.0
    sp_low     = np.array([1.0, 0.8, 0.5, 0.3], dtype=np.float32)  # min SP
    sp_high    = sp_low + 1.0  # e.g. [2.0,1.8,1.5,1.3] to avoid zero-range                                   # constant SP trajectories

    o_space = {
        "low":  np.concatenate([state_low,  sp_low]),
        "high": np.concatenate([state_high, sp_high]),
    }
    # Two pumps: u1 and u2
    a_space = {
        "low":  np.array([0.0, 0.0], dtype=np.float32),
        "high": np.array([1.0, 1.0], dtype=np.float32),
    }
    
    # Initial state: [h1, h2, h3, h4] + [SP1(0), SP2(0), SP3(0), SP4(0)]
    x0 = np.concatenate([
        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        np.array([SP["h1"][0], SP["h2"][0], SP["h3"][0], SP["h4"][0]], dtype=np.float32),
    ])

    params = {
       "N":       nsteps,
       "tsim":    tsim,
       "SP":      SP,
       "o_space": o_space,
       "model":   "four_tank",
       "a_space": a_space,
       "x0":      x0,
    }
    return pcgym.make_env(params)