# -*- coding: utf-8 -*-
"""
four_tank_env.py

Creates a PC-Gym environment for a four-tank process
with two manipulated variables (pump flows u1, u2).
https://maximilianb2.github.io/pc-gym/

This module defines your underlying plant model:

make_four_tank_env() builds a Gym-compatible four-tank process with two pumps and four interconnected water tanks.

It handles the hydraulic dynamics, set-point profiles (SP), normalization (which you’ve since disabled), and exposes the raw observation and action spaces.
"""

import numpy as np
import pcgym

nsteps = 100
tsim   = 25

# 1) Define SP and o_space at module‐level
SP = {
     "h1": np.ones(nsteps) * 1.0,
     "h2": np.ones(nsteps) * 0.8,
     "h3": np.ones(nsteps) * 0.5,
     "h4": np.ones(nsteps) * 0.3,
   }
o_space = {
    "low":  np.concatenate([np.zeros(4), np.array([1.0,0.8,0.5,0.3])]),
    "high": np.concatenate([np.ones(4)*2.0, np.array([2.0,1.8,1.5,1.3])]),
}

def make_four_tank_env(x0=None, nsteps=nsteps, tsim=tsim, force_init=False):

    # Setpoint trajectories for each tank (constant here)
    # Must match the model’s state names ["h1","h2","h3","h4"]

    # Two pumps: u1 and u2
    a_space = {
        "low": np.array([0.0, 0.0], dtype=np.float32),
        "high": np.array([10.0, 10.0], dtype=np.float32),
    }
    
    # randomize initial tank heights each episode between their o_space bounds
    # o_space.low/high for the first 4 entries are the h1..h4 bounds
    h_low, h_high = o_space["low"][:4], o_space["high"][:4]
    initial_h = np.random.uniform(low=h_low, high=h_high).astype(np.float32)
    # keep the set-points fixed
    initial_sp = np.array([
        SP["h1"][0],
        SP["h2"][0],
        SP["h3"][0],
        SP["h4"][0],
    ], dtype=np.float32)
    x0 = np.concatenate([initial_h, initial_sp])

    params = {
       "N":       nsteps,
       "tsim":    tsim,
       "SP":      SP,
       "o_space": o_space,
       "model":   "four_tank",
       "a_space": a_space,
       "x0":      x0,
       "normalize_o": False,    # disable the automatic [-1,+1] scaling
       "normalize_a": False,   # disable action scaling
    }
    env = pcgym.make_env(params)
    # Force-disable normalization at the object level
    env.normalize_o = False
    return env