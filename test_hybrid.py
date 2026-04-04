"""
Test script for the hybrid Street Fighter II agent.

Runs the HybridAgent (PPO + rule overrides) with visual rendering
and prints per-step diagnostics including which system (PPO vs rule)
chose each action.

Usage:
    python test_hybrid.py
    python test_hybrid.py --model ./opt/hybrid_final_model --episodes 3
    python test_hybrid.py --no-rules   # PPO-only baseline for comparison
"""

import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from hybrid_env import HybridStreetFighter
from hybrid_agent import HybridAgent


def test(model_path: str, num_episodes: int = 1, use_rules: bool = True):
    agent = HybridAgent(model_path, deterministic=True,
                        rule_override_enabled=use_rules)

    env = HybridStreetFighter(render_mode="human", reward_shaping=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")

    mode = "HYBRID (PPO + Rules)" if use_rules else "PPO ONLY"
    print(f"\nMode: {mode}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}\n")

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        agent.reset()
        done = False
        step_count = 0
        total_reward = 0

        print(f"=== Episode {episode} ===")

        while not done:
            # Extract info from the vectorized env's last step
            # On the first step after reset, info may not have game_state yet
            info = {}
            try:
                inner_env = env.envs[0].envs[0]  # unwrap DummyVecEnv→Monitor→HybridSF
                info = {
                    "health": inner_env.game_state.health,
                    "enemy_health": inner_env.game_state.enemy_health,
                    "score": inner_env.game_state.score,
                    "matches_won": inner_env.game_state.matches_won,
                    "enemy_matches_won": inner_env.game_state.enemy_matches_won,
                    "continuetimer": inner_env.game_state.continuetimer,
                }
            except (AttributeError, IndexError):
                pass

            action, rule_out = agent.predict(obs, info=info if info else None)
            obs, reward, done, vec_info = env.step(action)
            step_count += 1
            total_reward += reward[0]

            if reward[0] != 0:
                src = rule_out.rule_name or "ppo"
                print(f"  Step {step_count:5d} | {src:20s} | "
                      f"R={reward[0]:+7.1f} | Total={total_reward:8.1f} | "
                      f"HP={info.get('health', '?')}/{info.get('enemy_health', '?')}")

        print(f"--- Episode {episode} done ---")
        print(f"  Steps: {step_count}  Reward: {total_reward:.1f}")
        print()

    print("=" * 50)
    print(agent.get_stats_summary())
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hybrid SF2 agent")
    parser.add_argument("--model", type=str,
                        default="./opt/hybrid_trial_5",
                        help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--no-rules", action="store_true",
                        help="Disable rule overrides (PPO-only baseline)")
    args = parser.parse_args()

    test(model_path=args.model, num_episodes=args.episodes,
         use_rules=not args.no_rules)
