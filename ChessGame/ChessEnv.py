from typing import Optional
import random
# from ChessGame.game.abstract.action import AbstractActionFlags
from ChessGame.game.abstract.action import AbstractActionFlags
from ChessGame.game.abstract.piece import PieceColor
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np

from ChessGame.games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

# CHECKMATE_REWARD = 25_000
# CHECK_REWARD = 500
# INVALID_MOVE_REWARD = -1000

# Reward Constants (scaled to match max material ~125)
INVALID_MOVE_REWARD = -0.1      # Strong penalty for invalid moves
VALID_MOVE_REWARD = -0.001      # Small positive reward for valid move
# CHECK_REWARD = 50              # Check reward (≈ 40% of max material)
CHECKMATE_REWARD = 100          # Big reward for checkmate (≈ 4x max material)
DRAW_REWARD = 5               # Reward for draw (≈ material advantage)
MATERIAL_SCALE = 1 / 1000       # Keep material shaping (max ~125)
MAX_STEPS = 100

class MinichessEnv(gym.Env):
    def __init__(self, size: int = 5, invalid_action_masking = False, original_step = False) -> None:
        print(f"Initializing MiniChessEnv: {size}x{size} board with invalid action masking ={invalid_action_masking}, original_step={original_step}")
        self.original_step = original_step
        self.invalid_action_masking = invalid_action_masking
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.player = 1
        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        
        self.steps = 0
        self.action_space = Discrete(self.game.getActionSize())

        if(invalid_action_masking):
            # Masking invalid actions
            self.observation_space = Dict({
                "board": Box(-60000, 60000, shape=(5,5), dtype=np.float32),
            })
        else:
            self.observation_space = Dict({
                "board": Box(-60000, 60000, shape=(5,5), dtype=np.float32),
                "actions": Box(0, 1, shape=(self.action_space.n,), dtype=np.float32),
            })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.player = 1 #White: 1, Black: -1
        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        self.steps = 0
        return self._obs(), {}

    def step_original(self, action):
        if not action in self.legal_moves:
            print("action", action,"not in", self.legal_moves)
            # print([self.game.id_to_action[move] for move in self.legal_moves])
            print(self.game.display(self.board,self.player))
            print(self.player)
            print(self.game.id_to_action[action], "BAD ACTION ORIGINAL")
            assert False
            return self._obs(), -0.01, False, {}
        elif self.steps == 100:
            return self._obs(), -0.1, False, True, {}

        self.board, self.player = self.game.getNextState(self.board, self.player, action)
        reward = self.game.getGameEnded(self.board, 1)
        done = reward != 0
        
        # if done:
        #     print(self.steps)

        if not done:
            # Play random move for other agent
            legal_moves = list(self._get_legal_actions())
            move = random.choice(legal_moves)
            self.board, self.player = self.game.getNextState(self.board, self.player, move)
            reward = self.game.getGameEnded(self.board, 1)
            done = reward != 0

        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        obs = self._obs()

        reward = np.sum(obs["board"]) / 1000

        
        if done:
            # print(self.game.display(self.board, self.player))
            # print("\nGAME OVER {}\n".format(reward)) 
            pass

        self.steps += 1

        return obs, reward, done, False, {}
    
    def step(self, action):
        # if self.original_step:
        #     return self.step_original(action)

        self.steps += 1
        info = {}

        # -------------------------
        # 1️⃣ INVALID MOVE HANDLING
        # -------------------------
        if action not in self.legal_moves:
            if(self.invalid_action_masking):
                print("action", action,"not in", self.legal_moves)
                # print([self.game.id_to_action[move] for move in self.legal_moves])
                print(self.game.display(self.board,self.player))
                print(self.player)
                print(self.game.id_to_action[action], "BAD ACTION")
                assert self.invalid_action_masking, "Invalid action masking is enabled but action is not in legal moves."

            info["move"] = "invalid"
            info["chess_move"] = self.get_action_humanized(action, 1)
            reward = INVALID_MOVE_REWARD
            done = False  #It seems is better to not end the game on invalid move
            truncated = self.steps >= MAX_STEPS
            return self._obs(), reward, done, truncated, info

        # -------------------------
        # 2️⃣ PLAYER MOVE
        # -------------------------
        info["chess_move"] = self.get_action_humanized(action, 1)
        self.board, self.player = self.game.getNextState(self.board, self.player, action)
        reward = VALID_MOVE_REWARD  # Reward for making a valid move

        # Check game status after player's move
        game_result = self.game.getGameEnded(self.board, 1, 0.5)
        done = game_result != 0

        if done:
            # Terminal reward
            if game_result >= 1:  # Agent wins
                reward += CHECKMATE_REWARD
                info["result"] = "win"
            elif game_result < 0:  # Agent loses
                reward -= CHECKMATE_REWARD
                info["result"] = "loss"
            elif game_result == 0.5:  # Draw
                reward += DRAW_REWARD
                info["result"] = "draw"

        # # Add CHECK reward (only if game not finished)
        # if not done and AbstractActionFlags.CHECK in self.board.peek().modifier_flags:
        #     reward += CHECK_REWARD

        # -------------------------
        # 3️⃣ OPPONENT MOVE
        # -------------------------
        if not done:
            legal_moves = list(self._get_legal_actions())
            if legal_moves:
                move = random.choice(legal_moves)                
                self.board, self.player = self.game.getNextState(self.board, self.player, move)
                info["chess_move"] =  info["chess_move"] + "," + self.get_action_humanized(move, -1)
                game_result = self.game.getGameEnded(self.board, 1,0.5)
                done = game_result != 0

                if done:
                    if game_result >= 1:  # Opponent move leads to agent win
                        reward += CHECKMATE_REWARD
                        info["result"] = "win"
                    elif game_result < 0:  # Opponent move leads to agent loss
                        reward -= CHECKMATE_REWARD
                        info["result"] = "loss"
                    elif game_result == 0.5:  # Draw
                        reward += DRAW_REWARD
                        info["result"] = "draw"

                # # If opponent gave CHECK --> negative reward
                # if AbstractActionFlags.CHECK in self.board.peek().modifier_flags:
                #     reward -= CHECK_REWARD

        # -------------------------
        # 4️⃣ MATERIAL REWARD SHAPING
        # -------------------------
        obs = self._obs()
        reward += np.sum(obs["board"]) * MATERIAL_SCALE

        # -------------------------
        # 5️⃣ STEP MGMT
        # -------------------------
        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        # self.steps += 1
        truncated = self.steps >= MAX_STEPS

        info["move"] = "valid"
        return obs, reward, done, truncated, info    


    def get_action_humanized(self, action_id, player):                
       action = self.game.get_action_humanized(action_id, player)
       return action

    def _get_legal_actions(self, return_type="list"):
        legal_moves = self.game.getValidMoves(self.board, self.player, return_type=return_type)
        return set(legal_moves) if return_type == "list" else legal_moves
    
    # ------------------------------------------------------------------
    # Maskable PPO will call this every time it needs the action mask.
    # It MUST return a 1‑D boolean array of length == action_space.n
    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        # `legal_moves_one_hot` is already a 0/1 vector the same length
        # as the action space, so we just cast it to bool.
        return np.array(self.legal_moves_one_hot, dtype=bool)


    def get_legal_moves(self):
        return self._get_legal_actions(return_type="list")

    def _obs(self):
        if self.invalid_action_masking:
            board = np.array(self.board, dtype=np.float32)
            return {"board": board}
        
        board = np.array(self.board, dtype=np.float32)
        mask = np.array(self.legal_moves_one_hot, dtype=np.float32)
        return {"board": board, "actions": mask}

    def render(self, mode="human"):
        print("Current Board:")
        print(self.game.display(self.board, self.player))


def register_chess_env():
    if "gymnasium_env/ChessGame-v0" not in gym.envs.registry:
        gym.register(
            id="gymnasium_env/ChessGame-v0",
            entry_point="ChessGame.ChessEnv:MinichessEnv",
            max_episode_steps=300          
        )
