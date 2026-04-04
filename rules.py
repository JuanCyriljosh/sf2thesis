"""
Rule-based engine for Street Fighter II.

Evaluates the current GameState and returns:
  - strategy  : "aggressive" | "defensive" | "neutral"
  - override  : a concrete 12-button action array, or None to defer to PPO
  - reward_adj: an additive reward-shaping term

Rules are evaluated top-to-bottom; the first rule whose condition fires
produces the override (if any).  Strategy and reward shaping accumulate
from all matching rules.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from game_state import GameState


@dataclass
class RuleOutput:
    """Result returned by the rule engine each step."""
    strategy: str = "neutral"                       # high-level intent
    override: Optional[np.ndarray] = None           # action override (or None)
    reward_adjustment: float = 0.0                  # additive shaping reward
    rule_name: str = ""                             # which rule fired (debug)


# -----------------------------------------------------------------------
# Action builders  (MultiBinary(12) helpers)
# -----------------------------------------------------------------------
def _action(*buttons) -> np.ndarray:
    """Create a 12-dimensional binary action with the given buttons pressed."""
    a = np.zeros(12, dtype=np.int8)
    for b in buttons:
        a[b] = 1
    return a

GS = GameState  # alias for button constants

# Pre-built actions
ACTION_BLOCK_STANDING = _action(GS.BTN_LEFT)           # hold back = block
ACTION_BLOCK_CROUCHING = _action(GS.BTN_LEFT, GS.BTN_DOWN)
ACTION_LIGHT_PUNCH     = _action(GS.BTN_Y)
ACTION_HEAVY_PUNCH     = _action(GS.BTN_Z)
ACTION_HEAVY_KICK      = _action(GS.BTN_C)
ACTION_CROUCH_KICK     = _action(GS.BTN_DOWN, GS.BTN_B)
ACTION_JUMP_KICK       = _action(GS.BTN_UP, GS.BTN_C)
ACTION_FORWARD         = _action(GS.BTN_RIGHT)
ACTION_IDLE            = _action()

# Simple combo sequences (lists of actions to execute over consecutive frames)
COMBO_PRESSURE = [
    _action(GS.BTN_Y),                            # jab
    _action(GS.BTN_Y),                            # jab
    _action(GS.BTN_DOWN, GS.BTN_B),               # low kick
    _action(GS.BTN_Z),                            # heavy punch
]

COMBO_PUNISH = [
    _action(GS.BTN_Z),                            # heavy punch
    _action(GS.BTN_C),                            # heavy kick
]


class RuleEngine:
    """Stateful rule engine — call evaluate() once per step."""

    def __init__(self):
        self._combo_queue: list[np.ndarray] = []
        self._block_frames: int = 0                 # remaining frames to hold block

    def reset(self):
        self._combo_queue.clear()
        self._block_frames = 0

    def evaluate(self, state: GameState) -> RuleOutput:
        """Evaluate all rules against the current game state."""
        out = RuleOutput()

        # ----------------------------------------------------------
        # 0.  If a combo / block sequence is in progress, continue it
        # ----------------------------------------------------------
        if self._combo_queue:
            out.override = self._combo_queue.pop(0)
            out.rule_name = "combo_continuation"
            return out

        if self._block_frames > 0:
            self._block_frames -= 1
            out.override = ACTION_BLOCK_STANDING.copy()
            out.strategy = "defensive"
            out.rule_name = "block_continuation"
            return out

        # ----------------------------------------------------------
        # 1.  CRITICAL — emergency block when taking burst damage
        # ----------------------------------------------------------
        if state.damage_taken > 20:
            self._block_frames = 15          # hold block for 15 frames
            out.override = ACTION_BLOCK_CROUCHING.copy()
            out.strategy = "defensive"
            out.rule_name = "emergency_block"
            out.reward_adjustment = 0.5      # small reward for smart blocking
            return out

        # ----------------------------------------------------------
        # 2.  REWARD — successful block (chip damage detected)
        # ----------------------------------------------------------
        if state.blocked_hit:
            out.reward_adjustment += 1.0     # reward smart blocking
            out.strategy = "defensive"
            # No override — let PPO or a later rule decide the next move
            out.rule_name = "block_reward"
            # Don't return: let later rules also apply overrides if needed

        # ----------------------------------------------------------
        # 3.  REWARD — successful dodge + counter-hit (whiff punish)
        # ----------------------------------------------------------
        if state.successful_dodge:
            out.reward_adjustment += 2.0     # bigger reward for dodge → punish
            out.strategy = "aggressive"
            out.rule_name = "dodge_reward"
            # Don't return: let later rules stack overrides

        # ----------------------------------------------------------
        # 4.  DEFENSIVE — low health, play safe
        # ----------------------------------------------------------
        if state.player_low_health(0.25) and not state.enemy_low_health(0.25):
            out.strategy = "defensive"
            out.reward_adjustment = 0.3      # reward for surviving
            # Override only if also just took damage (pressure situation)
            if state.damage_taken > 0:
                self._block_frames = 10
                out.override = ACTION_BLOCK_STANDING.copy()
                out.rule_name = "low_health_block"
            else:
                out.rule_name = "low_health_caution"
            return out

        # ----------------------------------------------------------
        # 5.  AGGRESSIVE — enemy is low, go for the finish
        # ----------------------------------------------------------
        if state.enemy_low_health(0.20):
            out.strategy = "aggressive"
            out.reward_adjustment = 0.2
            # Queue a punish combo
            self._combo_queue = [a.copy() for a in COMBO_PUNISH]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "finish_combo"
            return out

        # ----------------------------------------------------------
        # 6.  ANTI-IDLE — force aggression when stalling
        # ----------------------------------------------------------
        if state.is_idle(patience=90):
            out.strategy = "aggressive"
            out.reward_adjustment = -0.5     # penalise idling
            out.override = ACTION_FORWARD.copy()
            out.rule_name = "anti_idle"
            return out

        # ----------------------------------------------------------
        # 7.  PUNISH — we just dealt damage, press advantage
        # ----------------------------------------------------------
        if state.damage_dealt > 15 and state.steps_since_damage_dealt == 0:
            out.strategy = "aggressive"
            self._combo_queue = [a.copy() for a in COMBO_PRESSURE]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "pressure_combo"
            return out

        # ----------------------------------------------------------
        # 8.  DEFAULT — let PPO decide
        # ----------------------------------------------------------
        out.strategy = "neutral"
        out.rule_name = "defer_to_ppo"
        return out
