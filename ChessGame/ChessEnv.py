from typing import Optional
import random
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np

from ChessGame.games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

class MinichessEnv(gym.Env):
    def __init__(self, size: int = 5) -> None:
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.player = 1
        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        
        self.steps = 0
        self.action_space = Discrete(self.game.getActionSize())
        self.observation_space = Dict({
            "board": Box(-60000, 60000, shape=(5,5), dtype=np.float32),
            "actions": Box(0, 1, shape=(self.action_space.n,), dtype=np.float32),
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.player = 1
        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        info = {}

        if action not in self.legal_moves:
            # Invalid action: penalize agent and end the game
            reward = -1.0
            done = True
            truncated = False
            info["result"] = "loss"  # Invalid move = loss
            return self._obs(), reward, done, truncated, info

        # Apply player's move
        self.board, self.player = self.game.getNextState(self.board, self.player, action)
        reward = self.game.getGameEnded(self.board, 1)
        done = reward != 0

        if done:
            # Assign result for PPO callback
            if reward > 0:
                info["result"] = "win"
            elif reward < 0:
                info["result"] = "loss"
            else:
                info["result"] = "draw"

        if not done:
            # Opponent plays random move
            legal_moves = list(self._get_legal_actions())
            move = random.choice(legal_moves)
            self.board, self.player = self.game.getNextState(self.board, self.player, move)
            reward = self.game.getGameEnded(self.board, 1)
            done = reward != 0
            if done:
                if reward > 0:
                    info["result"] = "win"
                elif reward < 0:
                    info["result"] = "loss"
                else:
                    info["result"] = "draw"

        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        obs = self._obs()

        # Optional shaping: tiny reward for board material
        reward = np.sum(obs["board"]) / 1000

        self.steps += 1
        truncated = self.steps >= 100

        return obs, reward, done, truncated, info


    def _get_legal_actions(self, return_type="list"):
        legal_moves = self.game.getValidMoves(self.board, self.player, return_type=return_type)
        return set(legal_moves) if return_type == "list" else legal_moves

    def get_legal_moves(self):
        return self._get_legal_actions(return_type="list")

    def _obs(self):
        board = np.array(self.board, dtype=np.float32)
        mask = np.array(self.legal_moves_one_hot, dtype=np.float32)
        return {"board": board, "actions": mask}

    def render(self, mode="human"):
        print("\nCurrent Board:")
        print(self.game.display(self.board, self.player))


def register_chess_env():
    if "gymnasium_env/ChessGame-v0" not in gym.envs.registry:
        gym.register(
            id="gymnasium_env/ChessGame-v0",
            entry_point="ChessGame.ChessEnv:MinichessEnv",
            max_episode_steps=300,
        )
