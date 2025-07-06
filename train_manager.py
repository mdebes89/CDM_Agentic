# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:08:57 2025

@author: mdbs1

The train_manager script is setting up and training an RL agent whose action space is which roles to turn on or off at each time step.

Objective: maximize long-term plant performance (keeping tank levels on their set-points) minus the cumulative cost of engaging each computational role.

Trade-off: engaging more roles usually gives better control proposals (higher perf_reward) but costs the manager more; engaging too few saves cost but risks poor plant behavior.

"""

from manager_env import HierarchicalManagerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from config import agentic


def make_sb3_env():
    """Vectorized & monitored env for SB3/PPO."""
    env = HierarchicalManagerEnv(debugging=False)
    return Monitor(env)


def main():
    train_env = DummyVecEnv([make_sb3_env])
    eval_env  = DummyVecEnv([make_sb3_env])
    
    # Callback that stops once the threshold is reached:
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)

    # Evaluate every 10000 steps:
    eval_cb = EvalCallback(eval_env, callback_after_eval=stop_cb,
                           eval_freq=10_000, n_eval_episodes=20,
                           verbose=1)

    model = PPO(
        "MlpPolicy",
        train_env,
        ent_coef=0.01,
        learning_rate=3e-4,
        batch_size=64,
        verbose=1,
    )
    model.learn(total_timesteps=200_000, callback=eval_cb)
    model.save("ppo_manager_tanks")
    
    obs = train_env.reset()
    
    for _ in range(50):
        action, _ = model.predict(obs)              # predict still returns (action, state)
        obs, rewards, dones, infos = train_env.step(action)
    
        # If that single env in the VecEnv signals done, reset (no unpacking)
        if dones[0]:
            obs = train_env.reset()
            
def test(num_steps: int = 100):
    """Run a fixed-length rollout with the trained manager and
    print per-step perf, cost, manager reward, plus final totals."""
    # 1) Re-instantiate env and load model
    env   = HierarchicalManagerEnv()
    model = PPO.load("ppo_manager_tanks")

    # 2) Reset
    obs, _ = env.reset()
    perf_list, cost_list, mgr_list = [], [], []

    # 3) Rollout loop
    for i in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, mgr_reward, terminated, truncated, info = env.step(action)

        perf = info["perf_reward"]
        cost = info["manager_cost"]

        perf_list.append(perf)
        cost_list.append(cost)
        mgr_list.append(mgr_reward)

        # print per-step metrics
        print(f"Step {i+1:3d}: perf={perf: .4f}, cost={cost: .4f}, manager={mgr_reward: .4f}")

        if terminated or truncated:
            obs, _ = env.reset()

    # 4) Summary totals
    test_type = "Pre-configured Agents"
    if agentic:
        test_type = "LLM Agents"    
    
    print(f"\nFinal totals using {test_type} over", num_steps, "steps:")
    print(f"  Total perf_reward    = {sum(perf_list): .4f}")
    print(f"  Total cost           = {sum(cost_list): .4f}")
    print(f"  Total manager_reward = {sum(mgr_list): .4f}")            

if __name__ == "__main__":
    main()
    print("\n=== TEST ROLLOUT ===")
    test(100)