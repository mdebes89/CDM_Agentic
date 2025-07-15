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
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from four_tank_tools import obs_tool, apply_action
from four_tank_env import make_four_tank_env
import numpy as np

from secrets import OPENAI_API_KEY


# No-op tool to capture the JSON action output
def set_action(json_str: str) -> str:
    # AgentExecutor will return this string as the tool result
    return json_str

# 1) Instantiate your LLM (e.g. GPT-4 via OpenAI)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)


# 2) Build the chat-style prompt
system_msg = SystemMessagePromptTemplate.from_template(
    """You control a PC-Gym Four-Tank process.
Each step you see six numbers [h1,h2,h3,h4,h3_SP,h4_SP].
You have one tool: set_action(json) which returns the JSON you pass it.

**Reference:** full environment spec here → https://maximilianb2.github.io/pc-gym/env/four_tank/

When you answer, you MUST output exactly:

Thought: <your reasoning>
Action: apply_action
Action Input: {"a1": <single-value float between 0.0 and 10.0>, "a2": <single-value float between 0.0 and 10.0>}

You proposed values must be a number values which is either 0 or a positive value up until 10.

Example (illustrative only; do not copy):

  Thought: h3 is slightly above its setpoint and h4 is slightly below, so adjust flows
  Action: apply_action
  Action Input: {"a1": 4.2, "a2": 5.8}

No other text, no headers, no “Final Answer:”.  
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
    tools=[apply_action],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    prompt=prompt,
    verbose=True,
    handle_parsing_errors=True,
)

def parse_actions(text: str) -> tuple[float, float]:
    # e.g. expect JSON: {"a1": 5.2, "a2": 3.1}
    import json
    data = json.loads(text)
    return data["a1"], data["a2"]

def train_episode(env, max_steps=200):
    # Reset env and clear agent memory at episode start
    obs, info = env.reset()
    total_reward = 0
    
    for _ in range(max_steps):       
        # a) Read the latest observation
        obs_dict = obs_tool.func(obs)
        # b) Build prompt with obs + cumulative reward
        # c) Ask the agent for its action
        # format only the HUMAN_SUFFIX with actual obs/reward
        user_input = (
        f"Observation: {obs_dict['h']}\n"
        f"Reward so far: {total_reward}\n"
        )
        response = manager_agent.run(user_input)
        a1, a2 = parse_actions(response)
        action = np.array([a1, a2], dtype=np.float32)
        # d) Apply that action in the env
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        if done:
            break
        
    return total_reward

if __name__ == "__main__":
    
    env = make_four_tank_env()
    reward = train_episode(env)
    print(f"Episode complete; total reward = {reward}")