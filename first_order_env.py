# -*- coding: utf-8 -*-
"""
cstr_env.py

Creates a PC-Gym environment for a continuously stirred-tank reactor (CSTR)
with a single manipulated variable (coolant temperature set-point).
https://maximilianb2.github.io/pc-gym/#quick-start
"""

import numpy as np
import pcgym

def make_first_order_env():
    # 1) Simulation horizon
    nsteps = 100
    tsim = 25

    # 2) Two-state setpoint trajectory: e.g. both variables ramp from 0→1
    SP = {
        "x1": list(np.linspace(0.2, 1.0, nsteps//2)) + list(np.linspace(1.0, 0.5, nsteps - nsteps//2)),
        "x2": list(np.linspace(0.5, 0.8, nsteps))
    }

    # 3) Observation space: [x1, x2, SP_x1, SP_x2]
    o_space = {
        "low":  np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "high": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    }

    # 4) Action space: assume two continuous inputs u1, u2
    a_space = {
        "low":  np.array([-1.0, -1.0], dtype=np.float32),
        "high": np.array([ 1.0,  1.0], dtype=np.float32),
    }

    # 5) Initial state
    x0 = np.array([0.2, 0.5, SP["x1"][0], SP["x2"][0]], dtype=np.float32)

    params = {
        "N":       nsteps,
        "tsim":    tsim,
        "SP":      SP,
        "o_space": o_space,
        "a_space": a_space,
        "x0":      x0,
        "model":   "first_order_system",  # swap here
    }

    return pcgym.make_env(params)