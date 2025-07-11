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

"""

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from four_tank_tools import obs_tool, act_tool
from pc_gym.env import FourTankEnv
from manager_env import ManagerEnv  # your orchestration from manager_env.py

# 1) Instantiate your LLM (e.g. GPT-4 via OpenAI)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 2) Create the agent with your tools
manager_agent = initialize_agent(
    tools=[obs_tool, act_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

def parse_actions(text: str) -> tuple[float, float]:
    # e.g. expect JSON: {"a1": 5.2, "a2": 3.1}
    import json
    data = json.loads(text)
    return data["a1"], data["a2"]

def train_episode(env, max_steps=200):
    manager_agent.memory.clear()
    total_reward = 0
    for _ in range(max_steps):
        # a) Read the latest observation
        obs = manager_agent.run("read_observation")
        # b) Build prompt with obs + cumulative reward
        prompt = (
            f"Observation: {obs['h']}\n"
            f"Reward so far: {env.current_reward()}\n"
            "Provide next action as JSON: {\"a1\": <0–10>, \"a2\": <0–10>}."
        )
        # c) Ask the agent for its action
        response = manager_agent.run(prompt)
        a1, a2 = parse_actions(response)
        # d) Apply that action in the env
        manager_agent.run(f"apply_action {a1} {a2}")
        total_reward = env.current_reward()
        if env.done:
            break
    return total_reward

if __name__ == "__main__":
    import gym
    from pc_gym.env import FourTankEnv

    env = gym.make("FourTank-v0")
    reward = train_episode(env)
    print(f"Episode complete; total reward = {reward}")