"""
Hybrid Agent — combines a trained PPO model with the rule-based engine.

Architecture
------------
                       ┌──────────────┐
  observation ───────► │  PPO Model   │──► ppo_action
                       └──────────────┘
                                           │
  info dict ──► GameState ──► RuleEngine ──┤
                                           │
                        ┌─────────────┐    ▼
                        │  Arbitrator  │◄──┘
                        └──────┬──────┘
                               │
                          final_action

The Arbitrator logic:
  1. If the rule engine returns an override  → use rule action
  2. Otherwise                               → use PPO action

This class is used at *inference / test time*.  During training, only
the reward-shaping component of the rule engine is active (inside
HybridStreetFighter).
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from game_state import GameState
from rules import RuleEngine, RuleOutput


class HybridAgent:
    """Wraps a PPO model and a RuleEngine for inference."""

    def __init__(self, model_path: str, deterministic: bool = True,
                 rule_override_enabled: bool = True):
        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        self.rule_override_enabled = rule_override_enabled

        self.game_state = GameState()
        self.rule_engine = RuleEngine()

        # Stats for analysis
        self.stats = {
            "total_steps": 0,
            "ppo_steps": 0,
            "rule_steps": 0,
            "rules_fired": {},
        }

    def reset(self):
        """Call at the start of each episode."""
        self.game_state.reset()
        self.rule_engine.reset()

    def predict(self, obs: np.ndarray, info=None):
        """
        Choose an action given the current observation and info dict.

        Parameters
        ----------
        obs   : stacked frame observation (from VecFrameStack)
        info  : the info dict from the environment step (contains RAM state)

        Returns
        -------
        action     : np.ndarray of shape (12,)
        rule_out   : RuleOutput with metadata about what the rule engine decided
        """
        # --- PPO prediction ---
        ppo_action, _states = self.model.predict(obs, deterministic=self.deterministic)

        # --- Rule evaluation ---
        rule_out = RuleOutput()
        if self.rule_override_enabled and info is not None:
            self.game_state.update(info)
            rule_out = self.rule_engine.evaluate(self.game_state)

        # --- Arbitration ---
        self.stats["total_steps"] += 1

        if rule_out.override is not None:
            action = rule_out.override
            # Reshape to match vectorized env expected shape if needed
            if ppo_action.ndim == 2:
                action = action.reshape(1, -1)
            self.stats["rule_steps"] += 1
            self.stats["rules_fired"][rule_out.rule_name] = (
                self.stats["rules_fired"].get(rule_out.rule_name, 0) + 1
            )
        else:
            action = ppo_action
            self.stats["ppo_steps"] += 1

        return action, rule_out

    def get_stats_summary(self) -> str:
        """Return a human-readable summary of rule vs PPO usage."""
        total = self.stats["total_steps"] or 1
        ppo_pct = 100 * self.stats["ppo_steps"] / total
        rule_pct = 100 * self.stats["rule_steps"] / total

        lines = [
            f"Total steps:  {self.stats['total_steps']}",
            f"PPO actions:  {self.stats['ppo_steps']} ({ppo_pct:.1f}%)",
            f"Rule actions: {self.stats['rule_steps']} ({rule_pct:.1f}%)",
            "Rules fired:",
        ]
        for name, count in sorted(self.stats["rules_fired"].items(),
                                   key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)
