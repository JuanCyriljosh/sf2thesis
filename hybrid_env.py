"""
Hybrid Street Fighter II environment.

Wraps the base StreetFighter env and adds:
  1. Rich reward shaping using GameState (health deltas, win/loss bonuses)
  2. Rule-engine reward adjustments (accumulated into the shaped reward)
  3. Game-state info forwarded in the info dict for the HybridAgent

During *training* this environment is used with a standard PPO model —
the rule engine only contributes reward shaping here.  Action overrides
happen in HybridAgent at *inference* time.
"""

import gymnasium as gym
from gymnasium.spaces import MultiBinary, Box

import numpy as np
import cv2
import pygame
import retro

from game_state import GameState
from rules import RuleEngine

NATIVE_W, NATIVE_H = 320, 224
SCALE = 3


class HybridStreetFighter(gym.Env):
    """Street Fighter II env with rule-based reward shaping."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None,
                 reward_shaping=True):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1),
                                    dtype=np.uint8)
        self.action_space = MultiBinary(12)

        self.game = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis",
            use_restricted_actions=retro.Actions.FILTERED,
        )

        # Rendering state
        self.rendering = (render_mode == "human")
        self._screen = None
        self._render_w = None
        self._render_h = None

        # Hybrid components
        self.game_state = GameState()
        self.rule_engine = RuleEngine()
        self.reward_shaping = reward_shaping

        # Reward-shaping weights (tunable)
        self.w_damage_dealt = 1.0      # reward per unit of damage dealt
        self.w_damage_taken = -0.5     # penalty per unit of damage taken
        self.w_win_round = 50.0        # bonus for winning a round
        self.w_lose_round = -30.0      # penalty for losing a round
        self.w_block = 3.0             # bonus for blocking an attack (chip damage)
        self.w_dodge = 5.0             # bonus for dodging + counter-hitting
        self.w_rule_adj = 1.0          # multiplier on rule-engine adjustments

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.game.reset()
        obs = self._preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        self.game_state.reset()
        self.rule_engine.reset()
        self._prev_matches_won = 0
        self._prev_enemy_matches_won = 0
        return obs, {}

    def step(self, action):
        obs, _reward, done, info = self.game.step(action)

        # Render if requested
        if self.rendering:
            self._render_frame(obs)

        # Preprocess observation
        obs = self._preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        # Update game state from RAM
        self.game_state.update(info)

        # Evaluate rules (for reward shaping — overrides applied externally)
        rule_out = self.rule_engine.evaluate(self.game_state)

        # ---- Compute reward ----
        if self.reward_shaping:
            reward = self._shaped_reward(info, rule_out)
        else:
            # Fall back to raw score delta (original behaviour)
            reward = info["score"] - self.score

        self.score = info["score"]

        # Enrich info dict so HybridAgent can read game state
        info["game_state"] = self.game_state.as_dict()
        info["rule_strategy"] = rule_out.strategy
        info["rule_name"] = rule_out.rule_name

        return frame_delta, reward, done, False, info

    def close(self):
        if self._screen is not None:
            pygame.quit()
        self.game.close()

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    def _shaped_reward(self, info: dict, rule_out) -> float:
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

        # 4. Rule-engine adjustment
        reward += self.w_rule_adj * rule_out.reward_adjustment

        return reward

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(observation: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resized, (84, 84, 1))

    def _render_frame(self, obs: np.ndarray):
        if self._screen is None:
            pygame.init()
            actual_h, actual_w, _ = obs.shape
            self._render_w = actual_w * SCALE
            self._render_h = actual_h * SCALE
            self._screen = pygame.display.set_mode(
                (self._render_w, self._render_h)
            )
            pygame.display.set_caption("Street Fighter II — Hybrid Agent")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        surface = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        scaled = pygame.transform.scale(surface,
                                        (self._render_w, self._render_h))
        self._screen.blit(scaled, (0, 0))
        pygame.display.flip()
