"""
Training script for the hybrid Street Fighter II agent.

Uses HybridStreetFighter (reward-shaped env) with PPO.
The rule engine contributes reward shaping during training;
action overrides are applied only at inference time via HybridAgent.

Usage:
    python train_hybrid.py                        # full Optuna HPO + final train
    python train_hybrid.py --skip-hpo             # train with default params
    python train_hybrid.py --timesteps 100000     # custom timestep budget
"""

import argparse
import os

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

from hybrid_env import HybridStreetFighter

LOG_DIR = "./logs/"
OPT_DIR = "./opt/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OPT_DIR, exist_ok=True)

# Default PPO hyperparameters (reasonable starting point for SF2)
DEFAULT_PARAMS = {
    "n_steps": 4096,
    "gamma": 0.94,
    "learning_rate": 5e-5,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
}


def make_env():
    """Create a fully wrapped hybrid environment."""
    env = HybridStreetFighter(reward_shaping=True)
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")
    return env


# ------------------------------------------------------------------
# Optuna hyperparameter search
# ------------------------------------------------------------------
def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_steps":       trial.suggest_int("n_steps", 2048, 8192, step=64),
        "gamma":         trial.suggest_float("gamma", 0.8, 0.9999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.4),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.8, 0.99),
    }


def objective(trial: optuna.Trial, timesteps: int) -> float:
    try:
        params = suggest_params(trial)
        env = make_env()

        model = PPO("CnnPolicy", env, tensorboard_log=LOG_DIR,
                     verbose=0, **params)
        model.learn(total_timesteps=timesteps)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()

        save_path = os.path.join(OPT_DIR, f"hybrid_trial_{trial.number}")
        model.save(save_path)
        print(f"Trial {trial.number}  mean_reward={mean_reward:.2f}  params={params}")

        return mean_reward

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -1000.0


def run_hpo(n_trials: int, timesteps: int) -> dict:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, timesteps),
                   n_trials=n_trials, n_jobs=1)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"  Value:  {study.best_trial.value:.2f}")
    print(f"  Params: {study.best_trial.params}")
    return study.best_trial.params


# ------------------------------------------------------------------
# Final training
# ------------------------------------------------------------------
def train_final(params: dict, timesteps: int):
    print(f"\nTraining final model for {timesteps} timesteps …")
    print(f"  Params: {params}")

    env = make_env()
    model = PPO("CnnPolicy", env, tensorboard_log=LOG_DIR,
                verbose=1, **params)
    model.learn(total_timesteps=timesteps)

    save_path = os.path.join(OPT_DIR, "hybrid_final_model")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Final eval  mean={mean_reward:.2f}  std={std_reward:.2f}")
    env.close()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train hybrid SF2 agent")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="Skip Optuna search, use default params")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of Optuna trials (default: 10)")
    parser.add_argument("--timesteps", type=int, default=30000,
                        help="Timesteps per training run (default: 30000)")
    parser.add_argument("--final-timesteps", type=int, default=None,
                        help="Timesteps for final training (default: same as --timesteps)")
    args = parser.parse_args()

    final_ts = args.final_timesteps or args.timesteps

    if args.skip_hpo:
        params = DEFAULT_PARAMS
        print("Skipping HPO — using default parameters")
    else:
        params = run_hpo(n_trials=args.n_trials, timesteps=args.timesteps)

    train_final(params, final_ts)


if __name__ == "__main__":
    main()
