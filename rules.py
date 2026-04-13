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

# -----------------------------------------------------------------------
# Ryu special move input sequences (frame-by-frame for retro emulator)
# -----------------------------------------------------------------------

# Hadouken (fireball): Down → Down-Forward → Forward + Punch
SPECIAL_HADOUKEN = [
    _action(GS.BTN_DOWN),
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),
    _action(GS.BTN_RIGHT, GS.BTN_Y),
]

# Shoryuken (dragon punch): Forward → Down → Down-Forward + Punch
SPECIAL_SHORYUKEN = [
    _action(GS.BTN_RIGHT),
    _action(GS.BTN_DOWN),
    _action(GS.BTN_DOWN, GS.BTN_RIGHT, GS.BTN_Z),
]

# Tatsumaki (hurricane kick): Down → Down-Back → Back + Kick
SPECIAL_TATSUMAKI = [
    _action(GS.BTN_DOWN),
    _action(GS.BTN_DOWN, GS.BTN_LEFT),
    _action(GS.BTN_LEFT, GS.BTN_C),
]

# -----------------------------------------------------------------------
# Combo sequences
# -----------------------------------------------------------------------

# Basic pressure: jab jab → low kick → heavy punch
COMBO_PRESSURE = [
    _action(GS.BTN_Y),                            # jab
    _action(GS.BTN_Y),                            # jab
    _action(GS.BTN_DOWN, GS.BTN_B),               # low kick
    _action(GS.BTN_Z),                            # heavy punch
]

# Simple punish: heavy punch → heavy kick
COMBO_PUNISH = [
    _action(GS.BTN_Z),                            # heavy punch
    _action(GS.BTN_C),                            # heavy kick
]

# Ryu bread-and-butter: crouching MK → Hadouken
COMBO_CR_MK_HADOUKEN = [
    _action(GS.BTN_DOWN, GS.BTN_A),               # crouching medium kick
    _action(GS.BTN_DOWN),                          # hadouken motion start
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            # diagonal
    _action(GS.BTN_RIGHT, GS.BTN_Y),               # fireball release
]

# Ryu punish combo: heavy punch → Shoryuken
COMBO_HP_SHORYUKEN = [
    _action(GS.BTN_Z),                            # heavy punch
    _action(GS.BTN_RIGHT),                         # shoryuken motion
    _action(GS.BTN_DOWN),
    _action(GS.BTN_DOWN, GS.BTN_RIGHT, GS.BTN_Z), # dragon punch release
]

# Ryu full confirm: jab jab → crouching MK → Hadouken
COMBO_FULL_CONFIRM = [
    _action(GS.BTN_Y),                            # jab
    _action(GS.BTN_Y),                            # jab
    _action(GS.BTN_DOWN, GS.BTN_A),               # crouching MK
    _action(GS.BTN_DOWN),                          # hadouken motion
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            # diagonal
    _action(GS.BTN_RIGHT, GS.BTN_Y),               # fireball release
]

# Jump-in combo: jumping HK → crouching HP → Hadouken
COMBO_JUMPIN = [
    _action(GS.BTN_UP, GS.BTN_RIGHT, GS.BTN_C),  # jumping heavy kick
    _action(),                                     # land frame
    _action(),                                     # land frame
    _action(GS.BTN_DOWN, GS.BTN_Z),               # crouching heavy punch
    _action(GS.BTN_DOWN),                          # hadouken motion
    _action(GS.BTN_DOWN, GS.BTN_RIGHT),            # diagonal
    _action(GS.BTN_RIGHT, GS.BTN_Y),               # fireball release
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
        #     Use Shoryuken combo for maximum damage finisher
        # ----------------------------------------------------------
        if (state.enemy_low_health(0.20)
                and state.damage_dealt > 0
                and not self._combo_queue):
            out.strategy = "aggressive"
            out.reward_adjustment = 5.0
            self._combo_queue = [a.copy() for a in COMBO_HP_SHORYUKEN]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "finish_shoryuken"
            return out

        # ----------------------------------------------------------
        # 6.  ZONING — at range with no recent action, throw fireball
        # ----------------------------------------------------------
        if (state.steps_since_damage_dealt > 60
                and state.steps_since_damage_taken > 60
                and not state.player_low_health(0.20)):
            out.strategy = "aggressive"
            out.reward_adjustment = 1.0
            self._combo_queue = [a.copy() for a in SPECIAL_HADOUKEN]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "zoning_hadouken"
            return out

        # ----------------------------------------------------------
        # 7.  ANTI-IDLE — penalise extreme stalling (once)
        # ----------------------------------------------------------
        if state.is_idle(patience=200) and not self._idle_penalised:
            out.strategy = "aggressive"
            out.reward_adjustment = -5.0
            out.rule_name = "anti_idle"
            self._idle_penalised = True
            return out

        # ----------------------------------------------------------
        # 8.  PUNISH — heavy hit landed, confirm into Hadouken combo
        # ----------------------------------------------------------
        if state.damage_dealt > 20 and state.steps_since_damage_dealt == 0:
            out.strategy = "aggressive"
            out.reward_adjustment = 3.0
            self._combo_queue = [a.copy() for a in COMBO_CR_MK_HADOUKEN]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "punish_hadouken"
            return out

        # ----------------------------------------------------------
        # 9.  PRESSURE — moderate hit landed, do full confirm combo
        # ----------------------------------------------------------
        if state.damage_dealt > 10 and state.steps_since_damage_dealt == 0:
            out.strategy = "aggressive"
            out.reward_adjustment = 2.0
            self._combo_queue = [a.copy() for a in COMBO_FULL_CONFIRM]
            out.override = self._combo_queue.pop(0)
            out.rule_name = "full_confirm"
            return out

        # ----------------------------------------------------------
        # 8.  DEFAULT — let PPO decide
        # ----------------------------------------------------------
        out.strategy = "neutral"
        out.rule_name = "defer_to_ppo"
        return out
