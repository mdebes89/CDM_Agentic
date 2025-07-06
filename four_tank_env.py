# -*- coding: utf-8 -*-
"""
four_tank_env.py

Creates a PC-Gym environment for a continuously stirred-tank reactor (CSTR)
with a single manipulated variable (coolant temperature set-point).
https://maximilianb2.github.io/pc-gym/#quick-start
"""

import numpy as np
import pcgym

def make_four_tank_env():
    nsteps = 100
    tsim   = 25.0
     # no SP trajectory for now, or define one per tank
    SP = {
        "x1": np.ones(nsteps)*1.0,
        "x2": np.ones(nsteps)*0.8,
        "x3": np.ones(nsteps)*0.5,
        "x4": np.ones(nsteps)*0.3,
    }
     # states = [h1,h2,h3,h4]
    o_space = {
        "low":  np.zeros(4, dtype=np.float32),
        "high": np.ones(4, dtype=np.float32)*2.0,
    }
     # two pumps, u1 and u2
    a_space = {
        "low":  np.array([0.0, 0.0], dtype=np.float32),
        "high": np.array([1.0, 1.0], dtype=np.float32),
    }
    
    x0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
     
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