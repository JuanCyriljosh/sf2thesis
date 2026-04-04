"""
Adaptive Agent — uses a trained PPO model that has LEARNED when to
defer to the rule engine via the gate bit.

Architecture
------------
                       +----------------+
  Dict observation --> | MultiInputPolicy|---> action (13 bits)
  (pixels + features)  |  (CNN + MLP)   |     bits 0-11: buttons
                       +----------------+     bit 12:    gate
                                                |
                              +-----------------+
                              |
                    AdaptiveStreetFighter.step()
                              |
              gate=1 + rule override?
              /                      \\
           YES                       NO
        execute rule               execute buttons

Unlike HybridAgent (which always overrides PPO when a rule fires),
this agent lets the LEARNED policy decide whether to follow rules.
The gate decision is part of the trained model's output.

This class is a thin wrapper for inference — the actual gate logic
lives inside AdaptiveStreetFighter.step().  The agent tracks
statistics about how often the learned policy chooses to defer.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO


class AdaptiveAgent:
    """Wraps a trained adaptive PPO model for inference."""

    def __init__(self, model_path, deterministic=True):
        self.model = PPO.load(model_path)
        self.deterministic = deterministic

        # Stats for analysis
        self.stats = {
            "total_steps": 0,
            "gate_activations": 0,   # agent set gate=1
            "gate_overrides": 0,     # gate=1 AND rule had override (from env info)
            "gate_no_override": 0,   # gate=1 but no rule override available
            "ppo_steps": 0,          # gate=0 (agent chose its own action)
            "rules_fired": {},
        }

    def reset(self):
        """Call at the start of each episode."""
        # Don't reset cumulative stats — they track across episodes.
        # Call reset_stats() explicitly if you want a clean slate.
        pass

    def reset_stats(self):
        """Reset all tracking statistics."""
        for key in self.stats:
            if isinstance(self.stats[key], dict):
                self.stats[key] = {}
            else:
                self.stats[key] = 0

    def predict(self, obs):
        """
        Choose an action given the current Dict observation.

        Parameters
        ----------
        obs : dict with "pixels" and "game_features" keys
              (as returned by AdaptiveStreetFighter)

        Returns
        -------
        action : np.ndarray of shape (13,) — pass directly to env.step()
        """
        action, _states = self.model.predict(obs, deterministic=self.deterministic)

        # Track gate usage from the model's output
        gate_bit = action[12] if action.ndim == 1 else action[0, 12]

        self.stats["total_steps"] += 1
        if gate_bit:
            self.stats["gate_activations"] += 1
        else:
            self.stats["ppo_steps"] += 1

        return action

    def update_stats(self, info):
        """
        Update stats from the env's info dict after stepping.

        Call this after env.step() to record whether the rule override
        was actually used (gate=1 and a rule was available).

        Parameters
        ----------
        info : dict returned by AdaptiveStreetFighter.step()
        """
        if info.get("used_rule", False):
            self.stats["gate_overrides"] += 1
            rule_name = info.get("rule_name", "unknown")
            self.stats["rules_fired"][rule_name] = (
                self.stats["rules_fired"].get(rule_name, 0) + 1
            )
        elif info.get("gate_active", False):
            # Gate was on but no rule override was available
            self.stats["gate_no_override"] += 1

    def get_stats_summary(self):
        """Return a human-readable summary of gate/rule/PPO usage."""
        total = max(1, self.stats["total_steps"])
        gate_pct = 100 * self.stats["gate_activations"] / total
        override_pct = 100 * self.stats["gate_overrides"] / total
        ppo_pct = 100 * self.stats["ppo_steps"] / total

        lines = [
            "=" * 50,
            "Adaptive Agent Statistics",
            "=" * 50,
            "Total steps:        {}".format(self.stats["total_steps"]),
            "",
            "PPO actions:        {} ({:.1f}%)".format(
                self.stats["ppo_steps"], ppo_pct),
            "Gate activations:   {} ({:.1f}%)".format(
                self.stats["gate_activations"], gate_pct),
            "  -> Rule used:     {} ({:.1f}%)".format(
                self.stats["gate_overrides"], override_pct),
            "  -> No override:   {}".format(
                self.stats["gate_no_override"]),
            "",
            "Rules fired:",
        ]

        if self.stats["rules_fired"]:
            for name, count in sorted(self.stats["rules_fired"].items(),
                                       key=lambda x: -x[1]):
                lines.append("  {}: {}".format(name, count))
        else:
            lines.append("  (none)")

        lines.append("=" * 50)
        return "\n".join(lines)
