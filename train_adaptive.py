"""
Training script for the adaptive Street Fighter II agent.

Uses AdaptiveStreetFighter with PPO's MultiInputPolicy.
The agent learns WHEN to defer to the rule engine via a gate bit,
rather than using hard-coded overrides.

The MultiInputPolicy automatically uses SB3's CombinedExtractor:
  - NatureCNN for the "pixels" observation  (84x84x4)
  - MLP       for the "game_features" obs   (12,)
  - Concatenated features → policy & value heads

Usage:
    python train_adaptive.py                        # full Optuna HPO + final train
    python train_adaptive.py --skip-hpo             # train with default params
    python train_adaptive.py --timesteps 500000     # custom timestep budget
"""

import argparse
import json
import os

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from adaptive_env import AdaptiveStreetFighter

LOG_DIR = "./logs_adaptive/"
OPT_DIR = "./opt_adaptive/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OPT_DIR, exist_ok=True)

# Default PPO hyperparameters (reasonable starting point)
DEFAULT_PARAMS = {
    "n_steps": 4096,
    "gamma": 0.95,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "ent_coef": 0.02,
    "batch_size": 128,
    "n_epochs": 5,
}


def make_env():
    """Create a single wrapped adaptive environment.

    Uses a factory function to avoid the closure-capture bug
    present in the original train_hybrid.py.  No VecFrameStack
    is needed — frame stacking is handled internally by the env.
    """
    def _init():
        env = AdaptiveStreetFighter(reward_shaping=True)
        return Monitor(env, LOG_DIR)
    return _init


# ------------------------------------------------------------------
# Optuna hyperparameter search
# ------------------------------------------------------------------
def suggest_params(trial):
    return {
        "n_steps":       trial.suggest_int("n_steps", 2048, 8192, step=1024),
        "gamma":         trial.suggest_float("gamma", 0.95, 0.999),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.3),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 0.99),
        "ent_coef":      trial.suggest_float("ent_coef", 0.01, 0.05),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs":      trial.suggest_categorical("n_epochs", [3, 5, 10]),
    }


def objective(trial, timesteps, n_envs):
    try:
        params = suggest_params(trial)
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        model = PPO("MultiInputPolicy", env, tensorboard_log=LOG_DIR,
                     verbose=0, **params)
        model.learn(total_timesteps=timesteps)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()

        save_path = os.path.join(OPT_DIR, "adaptive_trial_{}".format(trial.number))
        model.save(save_path)
        print("Trial {}  mean_reward={:.2f}  params={}".format(
            trial.number, mean_reward, params))

        return mean_reward

    except Exception as e:
        print("Trial {} failed: {}".format(trial.number, e))
        return -1000.0


def run_hpo(n_trials, timesteps, n_envs):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, timesteps, n_envs),
                   n_trials=n_trials, n_jobs=1)

    best = study.best_trial
    print("\nBest trial: {}".format(best.number))
    print("  Value:  {:.2f}".format(best.value))
    print("  Params: {}".format(best.params))

    # Save best params to file so they can be reviewed later
    results = {
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    results_path = os.path.join(OPT_DIR, "hpo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("  HPO results saved to {}".format(results_path))

    return best.params


# ------------------------------------------------------------------
# Final training
# ------------------------------------------------------------------
def train_final(params, timesteps, n_envs):
    print("\nTraining final adaptive model for {} timesteps ...".format(timesteps))
    print("  Params: {}".format(params))

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    model = PPO("MultiInputPolicy", env, tensorboard_log=LOG_DIR,
                verbose=1, **params)
    model.learn(total_timesteps=timesteps)

    save_path = os.path.join(OPT_DIR, "adaptive_final_model")
    model.save(save_path)
    print("Model saved to {}".format(save_path))

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
    print("Final eval  mean={:.2f}  std={:.2f}".format(mean_reward, std_reward))
    env.close()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train adaptive SF2 agent")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="Skip Optuna search, use default params")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of Optuna trials (default: 10)")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Timesteps per training run (default: 500000)")
    parser.add_argument("--final-timesteps", type=int, default=None,
                        help="Timesteps for final training (default: same as --timesteps)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    args = parser.parse_args()

    final_ts = args.final_timesteps or args.timesteps

    if args.skip_hpo:
        params = DEFAULT_PARAMS
        print("Skipping HPO - using default parameters")
    else:
        params = run_hpo(n_trials=args.n_trials, timesteps=args.timesteps,
                         n_envs=args.n_envs)

    train_final(params, final_ts, n_envs=args.n_envs)


if __name__ == "__main__":
    main()
