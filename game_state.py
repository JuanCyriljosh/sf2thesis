"""
Game state tracker for Street Fighter II.

Extracts RAM variables from the retro info dict and computes derived
features (health deltas, health ratios, match context) that the rule
engine uses for decision-making.
"""

import numpy as np


class GameState:
    """Tracks and derives game-state features from the retro info dict."""

    # ---- Genesis controller button indices (MultiBinary(12)) ----
    BTN_B     = 0   # Light Kick
    BTN_A     = 1   # Medium Kick
    BTN_MODE  = 2
    BTN_START = 3
    BTN_UP    = 4
    BTN_DOWN  = 5
    BTN_LEFT  = 6
    BTN_RIGHT = 7
    BTN_C     = 8   # Heavy Kick
    BTN_Y     = 9   # Light Punch
    BTN_X     = 10  # Medium Punch
    BTN_Z     = 11  # Heavy Punch

    # Convenience groups
    PUNCHES = (BTN_Y, BTN_X, BTN_Z)         # light, medium, heavy
    KICKS   = (BTN_B, BTN_A, BTN_C)         # light, medium, heavy
    ATTACKS = PUNCHES + KICKS

    # Health constants (SF2 Genesis default max health)
    MAX_HEALTH = 176

    # Chip damage threshold — blocked hits typically deal 1-5 damage
    # vs clean hits which deal 8-20+
    CHIP_DAMAGE_MAX = 5

    # Dodge detection window — how many frames after being hit we still
    # consider "active combat" for dodge detection
    DODGE_WINDOW = 15

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self):
        """Call on environment reset."""
        self.health = self.MAX_HEALTH
        self.enemy_health = self.MAX_HEALTH
        self.prev_health = self.MAX_HEALTH
        self.prev_enemy_health = self.MAX_HEALTH
        self.score = 0
        self.matches_won = 0
        self.enemy_matches_won = 0
        self.continuetimer = 0

        # Derived
        self.damage_dealt = 0          # this step
        self.damage_taken = 0          # this step
        self.health_ratio = 1.0        # player_health / max
        self.enemy_health_ratio = 1.0
        self.health_advantage = 0.0    # player - enemy  (normalised)
        self.is_winning = False
        self.is_losing = False
        self.round_over = False

        # Block / dodge detection
        self.blocked_hit = False         # True on frames where chip damage detected
        self.successful_dodge = False    # True on frames where dodge inferred
        self.total_blocks = 0
        self.total_dodges = 0

        # Tracking across episode
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.steps_since_damage_dealt = 0
        self.steps_since_damage_taken = 0
        self.step_count = 0

        # Combo tracking — consecutive hits within a short window
        self.combo_count = 0            # current combo length
        self.combo_window = 0           # frames since last hit in combo
        self.best_combo = 0             # longest combo this episode
        self.COMBO_WINDOW_MAX = 30      # max frames between hits to count as combo

    def update(self, info: dict):
        """Update state from the retro info dict. Call once per step."""
        self.step_count += 1

        # Store previous values
        self.prev_health = self.health
        self.prev_enemy_health = self.enemy_health

        # Read RAM values
        self.health = info.get("health", self.health)
        self.enemy_health = info.get("enemy_health", self.enemy_health)
        self.score = info.get("score", self.score)
        self.matches_won = info.get("matches_won", self.matches_won)
        self.enemy_matches_won = info.get("enemy_matches_won", self.enemy_matches_won)
        self.continuetimer = info.get("continuetimer", self.continuetimer)

        # Compute per-step deltas
        self.damage_dealt = max(0, self.prev_enemy_health - self.enemy_health)
        self.damage_taken = max(0, self.prev_health - self.health)

        # Cumulative tracking
        self.total_damage_dealt += self.damage_dealt
        self.total_damage_taken += self.damage_taken

        if self.damage_dealt > 0:
            self.steps_since_damage_dealt = 0
        else:
            self.steps_since_damage_dealt += 1

        if self.damage_taken > 0:
            self.steps_since_damage_taken = 0
        else:
            self.steps_since_damage_taken += 1

        # ---- Combo detection ----
        if self.damage_dealt > 0:
            if self.combo_window > 0 and self.combo_window <= self.COMBO_WINDOW_MAX:
                # Consecutive hit within window — extend combo
                self.combo_count += 1
            else:
                # First hit or gap too long — start new combo
                self.combo_count = 1
            self.combo_window = 0
            self.best_combo = max(self.best_combo, self.combo_count)
        else:
            self.combo_window += 1
            if self.combo_window > self.COMBO_WINDOW_MAX:
                self.combo_count = 0  # combo dropped

        # ---- Block detection ----
        # Chip damage (1 to CHIP_DAMAGE_MAX) indicates a blocked attack;
        # clean hits deal significantly more.
        self.blocked_hit = 0 < self.damage_taken <= self.CHIP_DAMAGE_MAX
        if self.blocked_hit:
            self.total_blocks += 1

        # ---- Dodge detection ----
        # A dodge is inferred when: (a) we took zero damage this frame,
        # (b) we were in active combat recently (took a hit within the
        #     last DODGE_WINDOW frames, so the enemy is still engaging),
        # (c) at least 1 frame has passed since the last hit (not the
        #     same frame we got hit).
        in_active_combat = (0 < self.steps_since_damage_taken
                            <= self.DODGE_WINDOW)
        self.successful_dodge = (self.damage_taken == 0
                                 and in_active_combat
                                 and self.damage_dealt > 0)
        if self.successful_dodge:
            self.total_dodges += 1

        # Ratios & advantage
        self.health_ratio = max(0, self.health) / self.MAX_HEALTH
        self.enemy_health_ratio = max(0, self.enemy_health) / self.MAX_HEALTH
        self.health_advantage = self.health_ratio - self.enemy_health_ratio
        self.is_winning = self.health_advantage > 0.1
        self.is_losing = self.health_advantage < -0.1

        # Round-over detection (both healths reset to max between rounds)
        both_full = (self.health >= self.MAX_HEALTH and
                     self.enemy_health >= self.MAX_HEALTH)
        self.round_over = both_full and self.step_count > 10

    # ------------------------------------------------------------------
    # Helpers used by rules
    # ------------------------------------------------------------------
    def player_low_health(self, threshold: float = 0.25) -> bool:
        return self.health_ratio <= threshold

    def enemy_low_health(self, threshold: float = 0.25) -> bool:
        return self.enemy_health_ratio <= threshold

    def is_idle(self, patience: int = 120) -> bool:
        """True if no damage dealt for `patience` steps."""
        return self.steps_since_damage_dealt >= patience

    def as_dict(self) -> dict:
        """Return a flat dict snapshot (useful for logging / debugging)."""
        return {
            "health": self.health,
            "enemy_health": self.enemy_health,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "health_ratio": round(self.health_ratio, 3),
            "enemy_health_ratio": round(self.enemy_health_ratio, 3),
            "health_advantage": round(self.health_advantage, 3),
            "matches_won": self.matches_won,
            "enemy_matches_won": self.enemy_matches_won,
            "steps_since_damage_dealt": self.steps_since_damage_dealt,
            "step_count": self.step_count,
            "blocked_hit": self.blocked_hit,
            "successful_dodge": self.successful_dodge,
            "total_blocks": self.total_blocks,
            "total_dodges": self.total_dodges,
            "combo_count": self.combo_count,
            "best_combo": self.best_combo,
        }
