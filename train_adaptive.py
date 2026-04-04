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
import os

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from adaptive_env import AdaptiveStreetFighter

LOG_DIR = "./logs_adaptive/"
OPT_DIR = "./opt_adaptive/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OPT_DIR, exist_ok=True)

# Default PPO hyperparameters (reasonable starting point)
DEFAULT_PARAMS = {
    "n_steps": 4096,
    "gamma": 0.94,
    "learning_rate": 5e-5,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
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
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "clip_range":    trial.suggest_float("clip_range", 0.15, 0.3),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 0.99),
        "ent_coef":      trial.suggest_float("ent_coef", 0.005, 0.02),
    }


def objective(trial, timesteps):
    try:
        params = suggest_params(trial)
        env = DummyVecEnv([make_env()])

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


def run_hpo(n_trials, timesteps):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, timesteps),
                   n_trials=n_trials, n_jobs=1)

    print("\nBest trial: {}".format(study.best_trial.number))
    print("  Value:  {:.2f}".format(study.best_trial.value))
    print("  Params: {}".format(study.best_trial.params))
    return study.best_trial.params


# ------------------------------------------------------------------
# Final training
# ------------------------------------------------------------------
def train_final(params, timesteps):
    print("\nTraining final adaptive model for {} timesteps ...".format(timesteps))
    print("  Params: {}".format(params))

    env = DummyVecEnv([make_env()])
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
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of Optuna trials (default: 10)")
    parser.add_argument("--timesteps", type=int, default=120000,
                        help="Timesteps per training run (default: 30000)")
    parser.add_argument("--final-timesteps", type=int, default=None,
                        help="Timesteps for final training (default: same as --timesteps)")
    args = parser.parse_args()

    final_ts = args.final_timesteps or args.timesteps

    if args.skip_hpo:
        params = DEFAULT_PARAMS
        print("Skipping HPO - using default parameters")
    else:
        params = run_hpo(n_trials=args.n_trials, timesteps=args.timesteps)

    train_final(params, final_ts)


if __name__ == "__main__":
    main()
