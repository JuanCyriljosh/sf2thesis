"""
Adaptive Street Fighter II environment.

Unlike HybridStreetFighter (which uses hard-coded rule overrides at
inference), this environment lets the PPO agent LEARN when to defer
to the rule engine via a 13th "gate" bit in the action space.

Architecture
------------
  Action = MultiBinary(13)
    bits 0-11 : standard Genesis controller buttons
    bit  12   : "defer-to-rule-engine" gate

  Observation = Dict
    "pixels"        : (84, 84, 4) — 4 stacked grayscale frames
    "game_features" : (12,)       — game state + rule context vector

Gate logic (inside step):
  if gate == 1 AND rule engine has an override:
      execute the rule engine's action
  else:
      execute the agent's 12 button bits

The agent learns through experience which situations benefit from
rule-based overrides vs its own learned policy.  Reward shaping is
always active regardless of gate state — only the action is gated.
"""

from collections import deque

import gymnasium as gym
from gymnasium.spaces import MultiBinary, Box, Dict

import numpy as np
import cv2
import pygame
import retro

from game_state import GameState
from rules import RuleEngine

NATIVE_W, NATIVE_H = 320, 224
SCALE = 3

# Number of scalar features exposed alongside pixels
N_GAME_FEATURES = 12

# Number of frames to stack in the pixel observation
N_FRAME_STACK = 4


class AdaptiveStreetFighter(gym.Env):
    """SF2 env with a learned rule-engine gate."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, reward_shaping=True):
        super().__init__()

        # --- Observation space ---
        # Pixels: 4 stacked grayscale 84x84 frames (internal stacking)
        # Game features: 12-element vector with game state + rule context
        self.observation_space = Dict({
            "pixels": Box(low=0, high=255,
                          shape=(84, 84, N_FRAME_STACK), dtype=np.uint8),
            "game_features": Box(low=-1.0, high=1.0,
                                 shape=(N_GAME_FEATURES,), dtype=np.float32),
        })

        # --- Action space ---
        # 12 game buttons + 1 gate bit
        self.action_space = MultiBinary(13)

        self.game = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis",
            use_restricted_actions=retro.Actions.FILTERED,
        )

        # Internal frame stack (avoids VecFrameStack + Dict obs issues)
        self._frames = deque(maxlen=N_FRAME_STACK)

        # Rendering state
        self.rendering = (render_mode == "human")
        self._screen = None
        self._render_w = None
        self._render_h = None

        # Hybrid components
        self.game_state = GameState()
        self.rule_engine = RuleEngine()
        self.reward_shaping = reward_shaping

         # Reward-shaping weights
        # Key principle: offense reward > defense penalty (asymmetric)
        # so the agent stays encouraged to fight even while losing
        self.w_damage_dealt = 2.0      # strongly reward landing hits
        self.w_damage_taken = -0.7     # moderate penalty (less than reward)
        self.w_win_round = 100.0       # strong incentive to finish rounds
        self.w_lose_round = -40.0      # moderate — don't crush early learning
        self.w_block = 5.0             # reward defensive play
        self.w_dodge = 8.0             # reward skillful dodges
        self.w_combo = 3.0             # reward per hit in a combo (scales with length)
        self.w_rule_adj = 1.0

        # Gate tracking (per-episode)
        self._gate_activations = 0
        self._gate_overrides = 0     # gate=1 AND rule had override
        self._total_steps = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs = self.game.reset()
        frame = self._preprocess(raw_obs)

        # Fill frame stack with copies of the first frame
        self._frames.clear()
        for _ in range(N_FRAME_STACK):
            self._frames.append(frame)

        self.score = 0
        self.game_state.reset()
        self.rule_engine.reset()
        self._prev_matches_won = 0
        self._prev_enemy_matches_won = 0
        self._gate_activations = 0
        self._gate_overrides = 0
        self._total_steps = 0

        obs = {
            "pixels": self._stacked_pixels(),
            "game_features": self._build_features(rule_out=None),
        }
        return obs, {}

    def step(self, action):
        # Handle vectorized envs (DummyVecEnv passes shape (1,13))
        action = np.asarray(action).flatten()

        gate_bit = action[12]
        game_action = action[:12].astype(np.int8)

        # ----------------------------------------------------------
        # Evaluate rules BEFORE stepping (need override for gate)
        # Snapshot state in case gate is not activated
        # ----------------------------------------------------------
        saved_combo = list(self.rule_engine._combo_queue)
        saved_block = self.rule_engine._block_frames

        rule_out = self.rule_engine.evaluate(self.game_state)

        # ----------------------------------------------------------
        # Gate arbitration
        # ----------------------------------------------------------
        self._total_steps += 1
        used_rule = False

        if gate_bit == 1 and rule_out.override is not None:
            effective_action = rule_out.override
            self._gate_activations += 1
            self._gate_overrides += 1
            used_rule = True
        else:
            effective_action = game_action
            # Restore rule engine state — override was not consumed
            self.rule_engine._combo_queue = saved_combo
            self.rule_engine._block_frames = saved_block
            if gate_bit == 1:
                self._gate_activations += 1     # gate wanted rules, but none available

        # ----------------------------------------------------------
        # Step the game
        # ----------------------------------------------------------
        raw_obs, _reward, done, info = self.game.step(effective_action)

        if self.rendering:
            self._render_frame(raw_obs)

        # Preprocess and update frame stack
        frame = self._preprocess(raw_obs)
        self._frames.append(frame)

        # Update game state from RAM
        self.game_state.update(info)

        # ----------------------------------------------------------
        # Compute reward (reward shaping always active, independent of gate)
        # ----------------------------------------------------------
        if self.reward_shaping:
            reward = self._shaped_reward(info, rule_out)
        else:
            reward = info["score"] - self.score

        self.score = info["score"]

        # ----------------------------------------------------------
        # Build observation
        # ----------------------------------------------------------
        obs = {
            "pixels": self._stacked_pixels(),
            "game_features": self._build_features(rule_out),
        }

        # Enrich info dict for logging / analysis
        info["game_state"] = self.game_state.as_dict()
        info["rule_strategy"] = rule_out.strategy
        info["rule_name"] = rule_out.rule_name
        info["gate_active"] = bool(gate_bit)
        info["used_rule"] = used_rule
        info["gate_rate"] = self._gate_overrides / max(1, self._total_steps)

        return obs, reward, done, False, info

    def close(self):
        if self._screen is not None:
            pygame.quit()
        self.game.close()

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------
    def _stacked_pixels(self):
        """Concatenate the frame stack along the channel axis → (84,84,4)."""
        return np.concatenate(list(self._frames), axis=-1)

    def _build_features(self, rule_out):
        """
        Build the 12-element game-feature vector.

        Index  Feature                        Range
        -----  -----------------------------  ---------
          0    health_ratio                   [0, 1]
          1    enemy_health_ratio             [0, 1]
          2    health_advantage               [-1, 1]
          3    damage_dealt (normalised)      [0, 1]
          4    damage_taken (normalised)      [0, 1]
          5    steps_since_damage_dealt (n.)  [0, 1]
          6    steps_since_damage_taken (n.)  [0, 1]
          7    blocked_hit                    {0, 1}
          8    successful_dodge               {0, 1}
          9    rule_has_override              {0, 1}
         10    rule_strategy == aggressive    {0, 1}
         11    rule_strategy == defensive     {0, 1}
        """
        gs = self.game_state
        f = np.zeros(N_GAME_FEATURES, dtype=np.float32)

        f[0] = gs.health_ratio
        f[1] = gs.enemy_health_ratio
        f[2] = gs.health_advantage
        f[3] = min(gs.damage_dealt / gs.MAX_HEALTH, 1.0)
        f[4] = min(gs.damage_taken / gs.MAX_HEALTH, 1.0)
        f[5] = min(gs.steps_since_damage_dealt / 300.0, 1.0)
        f[6] = min(gs.steps_since_damage_taken / 300.0, 1.0)
        f[7] = float(gs.blocked_hit)
        f[8] = float(gs.successful_dodge)

        if rule_out is not None:
            f[9]  = float(rule_out.override is not None)
            f[10] = float(rule_out.strategy == "aggressive")
            f[11] = float(rule_out.strategy == "defensive")

        return f

    # ------------------------------------------------------------------
    # Reward shaping  (same formula as hybrid_env)
    # ------------------------------------------------------------------
    def _shaped_reward(self, info, rule_out):
        gs = self.game_state
        reward = 0.0

        # 1. Damage-based reward
        reward += self.w_damage_dealt * gs.damage_dealt
        reward += self.w_damage_taken * gs.damage_taken

        # 2. Round outcome bonuses
        matches_won = info.get("matches_won", 0)
        enemy_matches_won = info.get("enemy_matches_won", 0)

        if matches_won > self._prev_matches_won:
            reward += self.w_win_round
        if enemy_matches_won > self._prev_enemy_matches_won:
            reward += self.w_lose_round

        self._prev_matches_won = matches_won
        self._prev_enemy_matches_won = enemy_matches_won

        # 3. Block / dodge bonuses
        if gs.blocked_hit:
            reward += self.w_block
        if gs.successful_dodge:
            reward += self.w_dodge

        # 4. Combo bonus — scales with combo length to reward multi-hit sequences
        if gs.combo_count >= 2:
            reward += self.w_combo * gs.combo_count

        # 5. Rule-engine adjustment (always active, independent of gate)
        reward += self.w_rule_adj * rule_out.reward_adjustment

        return reward

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(observation):
        """Convert raw RGB frame to grayscale 84x84x1."""
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (84, 84, 1))

    def _render_frame(self, obs):
        if self._screen is None:
            pygame.init()
            actual_h, actual_w, _ = obs.shape
            self._render_w = actual_w * SCALE
            self._render_h = actual_h * SCALE
            self._screen = pygame.display.set_mode(
                (self._render_w, self._render_h)
            )
            pygame.display.set_caption("Street Fighter II — Adaptive Agent")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        surface = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surface,
                                        (self._render_w, self._render_h))
        self._screen.blit(scaled, (0, 0))
        pygame.display.flip()
