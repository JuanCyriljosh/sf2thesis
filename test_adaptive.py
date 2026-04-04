"""
Test script for the adaptive Street Fighter II agent.

Runs the AdaptiveAgent (PPO with learned rule-engine gating) with
visual rendering and prints per-step diagnostics showing whether
the agent chose to defer to rules or use its own policy.

Usage:
    python test_adaptive.py
    python test_adaptive.py --model ./opt_adaptive/adaptive_final_model --episodes 3
"""

import argparse

from stable_baselines3.common.vec_env import DummyVecEnv

from adaptive_env import AdaptiveStreetFighter
from adaptive_agent import AdaptiveAgent


def test(model_path, num_episodes=1):
    agent = AdaptiveAgent(model_path, deterministic=True)

    # No VecFrameStack — frame stacking is handled inside the env
    env = AdaptiveStreetFighter(render_mode="human", reward_shaping=True)
    env = DummyVecEnv([lambda: env])

    print("\nMode: ADAPTIVE (Learned Gating)")
    print("Model: {}".format(model_path))
    print("Episodes: {}\n".format(num_episodes))

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        agent.reset()
        done = False
        step_count = 0
        total_reward = 0

        print("=== Episode {} ===".format(episode))

        while not done:
            action = agent.predict(obs)
            obs, reward, done, vec_info = env.step(action)

            # Update agent stats from env info
            info = vec_info[0]
            agent.update_stats(info)

            step_count += 1
            total_reward += reward[0]

            if reward[0] != 0:
                gate = "RULE" if info.get("used_rule", False) else "PPO"
                rule_name = info.get("rule_name", "")
                gs = info.get("game_state", {})
                hp = gs.get("health", "?")
                enemy_hp = gs.get("enemy_health", "?")

                print("  Step {:5d} | {:4s} | {:20s} | "
                      "R={:+7.1f} | Total={:8.1f} | "
                      "HP={}/{}".format(
                          step_count, gate, rule_name,
                          reward[0], total_reward,
                          hp, enemy_hp))

        print("--- Episode {} done ---".format(episode))
        print("  Steps: {}  Reward: {:.1f}".format(step_count, total_reward))
        print()

    print(agent.get_stats_summary())
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test adaptive SF2 agent")
    parser.add_argument("--model", type=str,
                        default="./opt_adaptive/adaptive_trial_4",
                        help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    args = parser.parse_args()

    test(model_path=args.model, num_episodes=args.episodes)
