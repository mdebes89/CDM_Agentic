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
from langchain.prompts import PromptTemplate
from four_tank_tools import obs_tool
from pc_gym.env import FourTankEnv


# 1) Instantiate your LLM (e.g. GPT-4 via OpenAI)
llm = ChatOpenAI(model="gpt-4", temperature=0)


# 2) Create the agent with your tools, *plus* a system/prefix that embeds
#    the env specs, role costs, tool API, and one tiny example.


# -- The system "prefix" every single call will begin with:
SYSTEM_PREFIX = """
You control a PC-Gym Four-Tank process.
- Obs vector: [h1,h2,h3,h4,h3_SP,h4_SP] ∈ [0.2,0.6] m
- Action: {"a1":0–10, "a2":0–10}
- Reward each step = –(|h3–h3_SP| + |h4–h4_SP|) – Σ(role_costs)
- Deadband on h3,h4: ±5% of setpoint (≈±0.01–0.03 m)
- Role costs: validator_h3=0.1, validator_h4=0.1,
  actionizer_h3=0.2, actionizer_h4=0.2,
  conditional=0.05, aggregator=0.05
Tools:
- read_observation(env) → {{"h":[...]} }
- apply_action(env,a1,a2) → {{"obs":[...],"reward":float,"done":bool}}

Example:
Observation: [0.45,0.47,0.30,0.32,0.30,0.30]
Reward so far: –0.02
→ {{ "a1": 4.8, "a2": 5.2 }}

Respond **only** with the next action JSON, no extra text.
"""

# We use LangChain's built-in zero-shot template slots:
PREFIX_TEMPLATE = SYSTEM_PREFIX + "\nObservation: {obs}\nReward so far: {total_reward}\n→"

prompt = PromptTemplate(
    template=PREFIX_TEMPLATE,
    input_variables=["obs","total_reward"],
)

manager_agent = initialize_agent(
    tools=[obs_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": prompt.template,
        "format_instructions": prompt.partial_format({"obs":"{{obs}}","total_reward":"{{total_reward}}"}),
    },
)

def parse_actions(text: str) -> tuple[float, float]:
    # e.g. expect JSON: {"a1": 5.2, "a2": 3.1}
    import json
    data = json.loads(text)
    return data["a1"], data["a2"]

def train_episode(env, max_steps=200):
    # Reset env and clear agent memory at episode start
    obs, info = env.reset()
    manager_agent.memory.clear()
    total_reward = 0
    
    for _ in range(max_steps):       
        # a) Read the latest observation
        obs_dict = obs_tool.func(env)
        # b) Build prompt with obs + cumulative reward
        prompt = (
            f"Observation: {obs_dict['h']}\n"
            f"Reward so far: {total_reward}\n"
            'Provide next action as JSON: {"a1": <0–10>, "a2": <0–10>}.'
        )
        # c) Ask the agent for its action
        response = manager_agent.run(prompt)
        a1, a2 = parse_actions(response)
        # d) Apply that action in the env
        obs, reward, done, info = env.step([a1, a2])
        total_reward += reward
        if done:
            break
        
    return total_reward

if __name__ == "__main__":
    import gym
    from pc_gym.env import FourTankEnv

    env = gym.make("FourTank-v0")
    reward = train_episode(env)
    print(f"Episode complete; total reward = {reward}")