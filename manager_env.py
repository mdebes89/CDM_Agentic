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
from tools.role_tools import perception_tool, planning_tool, control_tool

class ManagerEnv:
    def __init__(self, env):
        self.env = env
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        # Register all possible roles
        self.tools = [perception_tool, planning_tool, control_tool]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )

    def step(self):
        # 1) Get raw observation & reward
        raw_obs = self.env.get_observation()  
        reward = self.env.current_reward()

        # 2) Ask manager which roles to activate
        role_prompt = (
            f"Observation: {raw_obs}\n"
            f"Reward: {reward}\n"
            "Select which roles to run this step from [perception_role, planning_role, control_role].\n"
            "Return JSON: {\"roles\": [<role_names>] }."
        )
        role_response = self.agent.run(role_prompt)
        selected = json.loads(role_response)["roles"]

        # 3) Execute each selected role in sequence, feeding outputs forward
        data = raw_obs
        for role_name in selected:
            tool = next(t for t in self.tools if t.name == role_name)
            out = tool.run(data)
            data = {**data, **out}  # merge inputs & outputs

        # 4) After roles run, expect final 'action' in data
        action = data.get("action")
        if action is None:
            raise ValueError("No action produced by roles.")
        
        # 5) Apply action to env
        self.env.step(action)
        return action, reward

    def run_episode(self, max_steps=200):
        total = 0
        for _ in range(max_steps):
            a, r = self.step()
            total = self.env.current_reward()
            if self.env.done:
                break
        return total


if __name__ == "__main__":
    import gym
    from pc_gym.env import FourTankEnv

    env = gym.make("FourTank-v0")
    mgr = ManagerEnv(env)
    tot = mgr.run_episode()
    print(f"Episode complete. Total reward: {tot}")