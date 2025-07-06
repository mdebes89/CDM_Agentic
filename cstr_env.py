# -*- coding: utf-8 -*-
"""
cstr_env.py

Creates a PC-Gym environment for a continuously stirred-tank reactor (CSTR)
with a single manipulated variable (coolant temperature set-point).
https://maximilianb2.github.io/pc-gym/#quick-start
"""

import numpy as np
import pcgym

def make_cstr_env():
    # 1. Simulation horizon and timing
    nsteps = 100
    tsim   = 25

    # 2. Setpoint trajectory for C_A
    SP = {
        "Ca": [0.85] * (nsteps // 2) + [0.90] * (nsteps - nsteps // 2)
    }

    # 3. Define the observation space (Ca, T, Ca_SP)
    o_space = {
        "low":  np.array([0.7, 300.0, 0.7], dtype=np.float32),
        "high": np.array([1.0, 350.0, 1.0], dtype=np.float32),
    }

    # 4. Define the action space (single control: coolant temperature set-point)
    a_space = {
      "low":  np.array([295.0, 0.0], dtype=np.float32),
      "high": np.array([302.0, 1.0], dtype=np.float32),
    }
    # 5. Initial state [Ca, T, Ca_SP]
    x0 = np.array([0.8, 330.0, 0.8], dtype=np.float32)

    params = {
        "N":       nsteps,
        "tsim":    tsim,
        "SP":      SP,
        "o_space": o_space,
        "a_space": a_space,
        "x0":      x0,
        "model":   "cstr",   # two‐input ODE‐based variant in quick-start
    }

    return pcgym.make_env(params)