# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:08:57 2025

@author: mdbs1

Train the LLM-based manager to choose which roles to engage each timestep.

Objective: 
  Maximize long-run control performance (keeping h3,h4 at setpoints)
  minus computational cost of engaging each role.
  
Trade-off: engaging more roles usually gives better control proposals (higher perf_reward) but costs the manager more; engaging too few saves cost but risks poor plant behavior.

Four-tank env. spec: observation = [h1,h2,h3,h4, h3_SP,h4_SP], action ∈ [0,10]² : https://maximilianb2.github.io/pc-gym/env/four_tank/

Typical tank heights span 0.2–0.6 m, so a 5% band (≈0.01–0.03 m) is a reasonable deadband: https://arxiv.org/html/2410.22093v2

High-Level Architecture
Environment wrapper (four_tank_env.py)

Builds a PC-Gym Four-Tank environment with two pumps and four interconnected water tanks.

Disables the built-in normalization and fixes constant set-points at module load.

Primitive tools (four_tank_tools.py)

read_observation(env) → {"h": […]}

apply_action(env, a1, a2) → {"status": "applied"}

Hierarchical manager (manager_env.py + executor.py)

Roles (validator_h3/h4, actionizer_h3/h4, conditional_role, aggregate_actions) implemented with deterministic GPT-4 prompts.

Manager agent (LangChain ZERO_SHOT_REACT_DESCRIPTION) chooses a subset of roles each step, invokes them in sequence, and issues the final action to the plant.

Training script (train_manager.py)

A simpler two-tool agentic loop (observe → prompt LLM → parse JSON → apply → repeat) intended to “train” the manager, though no learning algorithm is actually employed.

"""

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import numpy as np, json, re
from four_tank_env import make_four_tank_env
from four_tank_tools import obs_tool
from secrets import OPENAI_API_KEY


# 1) Instantiate your LLM (e.g. GPT-4 via OpenAI)
llm = ChatOpenAI(
    model="gpt-4-0613",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY,
)

    
PROMPT_TEMPLATE = """
You are the four-tank control manager for the PC-Gym `four_tank` environment.
(Ref: https://maximilianb2.github.io/pc-gym/env/four_tank/)

DEFINITIONS:
  d3 = h3 - 0.50
  d4 = h4 - 0.30
  prev_d3 = previous_h3 - 0.50
  prev_d4 = previous_h4 - 0.30

CONTROL COUPLING (assumed):
  a1 ↑ ⇒ h3 ↑ (primary), slight h4 effect.
  a2 ↑ ⇒ h4 ↑ (primary), slight h3 effect.

DEADBAND: db = 0.02 (|d3| ≤ db and |d4| ≤ db considered “on target”).

CURRENT STATE:
  h1 = {h1:.4f}
  h2 = {h2:.4f}
  h3 = {h3:.4f}
  h4 = {h4:.4f}
  previous_action = {prev_action}   # [prev_a1, prev_a2]
  last_reward = {last_reward:.6f}
  total_reward = {total_reward:.6f}

CONTROL OUTPUT:
  Choose new pump settings a1, a2 in continuous range [0, 10].

PRIMARY OBJECTIVE:
  Drive h3 → 0 and h4 → 0 (reduce absolute deviations |h3| and |h4|).

SECONDARY OBJECTIVES / CONSTRAINTS:
  • Keep actions within [0,10]; if a suggested value is outside, saturate at boundary.
  • Avoid unnecessary large simultaneous changes (Δa1 and Δa2 both > 1.5) unless BOTH |h3| and |h4| are large (> 0.4).
  • Encourage *some* adjustment if last_reward ≤ 0 and we are not at boundaries.
  • Penalize repeating the exact previous action when last_reward ≤ 0 (must adjust at least one pump by ≥ 0.2).
  • If last_reward improved ( > 0 ), small refinements (|Δa| ≈ 0–0.3) are preferred.
  • Prefer smooth changes (|Δa| ≤ 0.8) unless there is a large deviation (|h3| or |h4| > 0.7).

DIRECTIONAL HEURISTICS:
  • If h3 > +0.1 (too high) → decrease a1 modestly (0.2–0.6). If h3 < -0.1 (too low) → increase a1 (0.2–0.6).
  • If h4 > +0.1 → decrease a2. If h4 < -0.1 → increase a2.
  • If both h3 and h4 are within a small deadband (|h3|, |h4| ≤ 0.05) → hold or make very small stabilizing tweaks (≤ 0.2).
  • If h3 deviation magnitude >> h4 (|h3| ≥ |h4| + 0.15) bias more change in a1 than a2, and vice versa.

TIE-BREAKING / SAFETY:
  • If unsure, bias toward small corrective moves rather than 0 change.
  • Never output prose or explanation.
  • The action must not be identical to the previous action when last_reward ≤ 0 (unless both pumps are at a boundary and change is impossible).
  • If last_reward < 0 and you repeat previous_action exactly, that is penalized.

OUTPUT FORMAT (NO EXTRA TEXT):
[a1, a2]

DO NOT output explanations, bullet points, or words.
Invalid: "I propose [4.2, 5.1]"   (contains words)
Valid: [4.2, 5.1]

REMEMBER: Output ONLY the two pump settings via the function call. No commentary.
"""

ARRAY_REGEX = re.compile(r"\[\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*,\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*\]")

def decide_action(h_vec, last_reward, total_reward, prev_action):
    prompt = PROMPT_TEMPLATE.format(
        h1=h_vec[0], h2=h_vec[1], h3=h_vec[2], h4=h_vec[3],
        prev_action=list(prev_action),
        last_reward=last_reward,
        total_reward=total_reward
    )
    msg = llm.predict_messages([HumanMessage(content=prompt)])
    txt = (msg.content or "").strip()
    m = ARRAY_REGEX.search(txt)
    if not m:
        # fallback: try to extract any two numbers
        nums = re.findall(r"[-+]?\d*\.?\d+", txt)
        if len(nums) >= 2:
            a1, a2 = float(nums[0]), float(nums[1])
        else:
            a1, a2 = prev_action  # fallback: keep previous
    else:
        a1, a2 = json.loads(m.group(0))
    # clamp + minimal exploration if no change but reward bad
    a1 = min(10.0, max(0.0, float(a1)))
    a2 = min(10.0, max(0.0, float(a2)))
    if last_reward <= 0 and abs(a1 - prev_action[0]) < 0.15 and abs(a2 - prev_action[1]) < 0.15:
        # force slight exploration
        a1 = min(10, max(0, a1 + np.random.uniform(-0.4, 0.4)))
        a2 = min(10, max(0, a2 + np.random.uniform(-0.4, 0.4)))
    return a1, a2


def train_episode(env, max_steps=200):
    obs, _ = env.reset()
    total_reward = 0.0
    last_reward = 0.0
    prev_action = (5.0, 5.0)
    for step in range(max_steps):
        obs_dict = obs_tool.func(obs)
        h_vec = obs_dict["h"]
        a1, a2 = decide_action(h_vec, last_reward, total_reward, prev_action)
        action = np.array([a1, a2], dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"Step {step+1:03d}: h={np.round(h_vec,3)} act=({a1:.2f},{a2:.2f}) "
              f"Δ=({a1-prev_action[0]:+.2f},{a2-prev_action[1]:+.2f}) r={reward:.4f}")
        total_reward += reward
        last_reward = reward
        prev_action = (a1, a2)
        if terminated or truncated:
            break
    return total_reward

if __name__ == "__main__":
    
    env = make_four_tank_env()
    reward = train_episode(env)
    print(f"Episode complete; total reward = {reward}")