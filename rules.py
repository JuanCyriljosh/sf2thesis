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
        self._block_frames: int = 0
        self._idle_penalised: bool = False   # prevent anti-idle spam

    def reset(self):
        self._combo_queue.clear()
        self._block_frames = 0
        self._idle_penalised = False

    def evaluate(self, state: GameState) -> RuleOutput:
        """Evaluate all rules against the current game state."""
        out = RuleOutput()

        # Reset idle flag once the agent does something
        if state.damage_dealt > 0:
            self._idle_penalised = False

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
        # 1.  EMERGENCY — block only on heavy burst damage
        # ----------------------------------------------------------
        if state.damage_taken > 30:
            self._block_frames = 8
            out.override = ACTION_BLOCK_CROUCHING.copy()
            out.strategy = "defensive"
            out.rule_name = "emergency_block"
            out.reward_adjustment = 3.0
            return out

        # ----------------------------------------------------------
        # 2.  REWARD — successful block (chip damage detected)
        # ----------------------------------------------------------
        if state.blocked_hit:
            out.reward_adjustment += 5.0
            out.strategy = "defensive"
            out.rule_name = "block_reward"
            # No override, no return — let later rules also check

        # ----------------------------------------------------------
        # 3.  REWARD — dodge + counter-hit (whiff punish)
        # ----------------------------------------------------------
        if state.successful_dodge:
            out.reward_adjustment += 8.0
            out.strategy = "aggressive"
            out.rule_name = "dodge_reward"
            # No override, no return — let later rules also check

        # ----------------------------------------------------------
        # 4.  DEFENSIVE — low health and under pressure, block
        # ----------------------------------------------------------
        if (state.player_low_health(0.20)
                and not state.enemy_low_health(0.20)
                and state.damage_taken > 0):
            out.strategy = "defensive"
            out.reward_adjustment = 3.0
            self._block_frames = 8
            out.override = ACTION_BLOCK_STANDING.copy()
            out.rule_name = "low_health_block"
            return out

        # ----------------------------------------------------------
        # 5.  AGGRESSIVE — enemy low + we just landed a hit, finish
        # ----------------------------------------------------------
        if (state.enemy_low_health(0.20)
                and state.damage_dealt > 0
                and not self._combo_queue):
            out.strategy = "aggressive"
            out.reward_adjustment = 5.0
            self._combo_queue = [a.copy() for a in COMBO_PUNISH]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "finish_combo"
            return out

        # ----------------------------------------------------------
        # 6.  ANTI-IDLE — penalise extreme stalling (once)
        # ----------------------------------------------------------
        if state.is_idle(patience=200) and not self._idle_penalised:
            out.strategy = "aggressive"
            out.reward_adjustment = -5.0
            out.rule_name = "anti_idle"
            self._idle_penalised = True
            return out

        # ----------------------------------------------------------
        # 7.  PUNISH — heavy hit landed, press advantage
        # ----------------------------------------------------------
        if state.damage_dealt > 20 and state.steps_since_damage_dealt == 0:
            out.strategy = "aggressive"
            out.reward_adjustment = 3.0
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
