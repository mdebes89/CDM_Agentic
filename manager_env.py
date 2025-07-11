# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:07:58 2025

@author: mdbs1

Wraps the four-tank plant with a hierarchical decision layer (the “manager”):

On each step it takes a 4-bit manager_action flag vector → decides which of your child roles to invoke (validators, actionizers, aggregator, etc.).

It collects their proposed actuator commands into a final_dict, flattens that to the 2-D action [u1,u2], steps the plant, then computes

It tracks debugging info so you can see which roles ran, what they proposed, and the resulting cost/performanc
"""

import json
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from config import ROLE_COSTS
from pcgym import gym

from executor import (
    validator_h3, validator_h4,
    actionizer_h3, actionizer_h4,
    conditional_role, aggregate_actions
)

# wrap each as a Tool
tools = [
    Tool.from_function(validator_h3, name="validator_h3",
                       description="Return yes/no if h3 error exceeds deadband"),
    Tool.from_function(validator_h4, name="validator_h4",
                       description="Return yes/no if h4 error exceeds deadband"),
    Tool.from_function(actionizer_h3, name="actionizer_h3",
                       description="Propose an adjustment for valve u1 based on h3 error"),
    Tool.from_function(actionizer_h4, name="actionizer_h4",
                       description="Propose an adjustment for valve u2 based on h4 error"),
    Tool.from_function(conditional_role, name="conditional_role",
                       description="Resolve conflicts among proposed adjustments"),
    Tool.from_function(aggregate_actions, name="aggregate_actions",
                       description="Finalize u1/u2 adjustment JSON"),
]

class ManagerEnv:
    def __init__(self, env):
        self.env = env
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )

    def step(self):
        # 1) Get raw observation & reward
        obs = self.env.get_observation()  # full 8-dim
        reward = self.env.current_reward()

        # 2) Ask manager which roles to activate
        prompt = (
            f"Observation: {obs.tolist()}\n"
            f"Reward: {reward}\n"
            "Which roles should run this step? "
            "Return JSON {\"roles\": [<tool_names>] }."
        )
        roles = json.loads(self.agent.run(prompt))["roles"]

        # 3) Execute each selected role in sequence, feeding outputs forward
        data = obs.tolist()
        for role in roles:
            out = next(t for t in tools if t.name == role).run(data)
            # if tool returns a dict, merge into data‐vector or keep track
            if isinstance(out, dict):
                data = {**{"h": data}, **out}

        # 4) After roles run, expect final 'action' in data
        action = data.get("u1_u2") or out.get("u1_u2")
        obs, raw_reward, done, info = self.env.step(action)
        # 5) subtract the cost of each role used
        role_cost = sum(ROLE_COSTS[r] for r in self.current_roles)
        net_reward = raw_reward - role_cost
        return action, net_reward, done


    def run_episode(self, max_steps: int = 1000):
        # 1) reset at episode start
        obs, info = self.env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action, reward, done = self.step()
            total_reward += reward       # 2) accumulate
            if done:
                break                   # 3) stop on terminal
        return total_reward


if __name__ == "__main__":

    env = gym.make("FourTank-v0")
    mgr = ManagerEnv(env)
    tot = mgr.run_episode()
    print(f"Episode complete. Total reward: {tot}")