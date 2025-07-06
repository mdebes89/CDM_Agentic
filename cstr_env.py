# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pcgym

def make_cstr_env():
    env = pcgym.make_env({"model": "cstr"})
    return env