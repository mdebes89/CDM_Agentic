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
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from four_tank_tools import obs_tool, set_action_tool
from four_tank_env import make_four_tank_env
import numpy as np
import json, re
from secrets import OPENAI_API_KEY




# 1) Instantiate your LLM (e.g. GPT-4 via OpenAI)
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)


# 2) Build the chat-style prompt
system_msg = SystemMessagePromptTemplate.from_template(
    """You control a PC-Gym Four-Tank process.
Each step you see six numbers [h1,h2,h3,h4,h3_SP,h4_SP].
You have two tools:
  • apply_action([x,y])   – steps the env and returns status.
  • set_action(json_str)  – captures your JSON and returns it.

**Reference:** full environment spec here → https://maximilianb2.github.io/pc-gym/env/four_tank/

When you choose your next action, **CALL ONLY** the `set_action` tool. Do NOT emit any other text.

Your tool call must look exactly like:

  Action: set_action
  Action Input: [<float1>, <float2>]

No markdown, no code fences, no extra prose.   
"""
)
    
# 3) Insert the scratchpad slot
ai_scratchpad = AIMessagePromptTemplate.from_template("{agent_scratchpad}")

    
# 4) Define the human slot for obs + reward
human_msg = HumanMessagePromptTemplate.from_template(
    "Observation: {h}\nReward so far: {reward}"
)


# 5) Combine into one ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    system_msg,
    ai_scratchpad,
    human_msg,
])

manager_agent = initialize_agent(
    tools=[set_action_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    prompt=prompt,
    verbose=False,
    handle_parsing_errors=True,
)

def parse_actions(text: str) -> tuple[float, float]:
    # text will be something like "[3.5,6.2]"
    a1, a2 = json.loads(text)
    return float(a1), float(a2)

def train_episode(env, max_steps=200):
    # 1) Read & format obs
    obs, info = env.reset()
    total_reward = 0
    last_reward = 0.0
    
    for _ in range(max_steps):       
        # 1) Read & format obs
        obs_dict = obs_tool.func(obs)
        # 2) Build prompt
        user_input = (
            f"Observation: {obs_dict['h']}\n"
            f"Last step reward: {last_reward}\n"
            f"Total reward: {total_reward}\n\n"
            "Example of desired output:\n"
            "{\n"
            "  \"action\": \"apply_action\",\n"
            "  \"action_input\": [2.5, 4.0]\n"
            "}\n"
        )
        # 3) Ask manager → calls set_action, returns e.g. "[3.2,5.7]"
        raw = manager_agent.run(user_input)
        # 4) Parse that array
        m = re.search(r"\[\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\]", raw)
        if not m:
            raise ValueError(f"No [a1,a2] found in:\n{raw!r}")
        a1, a2 = json.loads(m.group(0))
        a1, a2 = float(a1), float(a2)

        action = np.array([a1, a2], dtype=np.float32)
        # 5) Apply that action in the env
        obs, reward, terminated, truncated, info = env.step(action)
        # 6) Log and accumulate
        print(f"Step {_+1:3d}: action=({a1:.3f},{a2:.3f}), reward={reward:.6f}")
        done = terminated or truncated
        
        total_reward += reward
        last_reward = reward
        if done:
            break
        
    return total_reward

if __name__ == "__main__":
    
    env = make_four_tank_env()
    reward = train_episode(env)
    print(f"Episode complete; total reward = {reward}")