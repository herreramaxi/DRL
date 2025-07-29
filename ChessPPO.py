import os
from ChessGame.ChessEnv import register_chess_env
from cudaCheck import is_cuda_available
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torchinfo import summary

# ✅ Hyperparameters
MODEL_PATH = "ppo_chess.zip"
LOG_DIR = "./chess_logs"
TOTAL_TIMESTEPS = 1_000_000  # ✅ Increased for meaningful training
N_ENVS = 10  # ✅ Parallel envs for speed
N_STEPS = 2048  # ✅ More stable with PPO
BATCH_SIZE = 512  # ✅ Must divide n_steps * n_envs (2048 * 8 = 16384)
N_EPOCHS = 10  # ✅ PPO update passes


class WinRateCallback(BaseCallback):
    def __init__(self, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.wins, self.losses, self.draws = 0, 0, 0
        self.invalid_moves = 0
        self.valid_moves = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        # Check infos from vectorized env
        for info in self.locals.get("infos", []):
            if "move" in info:
                if info["move"] == "invalid":
                   self.invalid_moves += 1
                else:
                     self.valid_moves += 1

            if "result" in info:  # Our env provides result
                self.episodes += 1
                if info["result"] == "win":
                    self.wins += 1
                elif info["result"] == "loss":
                    self.losses += 1
                else:
                    self.draws += 1

        if self.num_timesteps % self.log_interval == 0 and self.episodes > 0:
            win_rate = self.wins / self.episodes
            self.logger.record("custom/invalid_moves", self.invalid_moves)
            self.logger.record("custom/valid_moves", self.valid_moves)
            self.logger.record("custom/win_rate", win_rate)
            self.logger.record("custom/episodes", self.episodes)
            self.logger.record("custom/wins", self.wins)
            self.logger.record("custom/losses", self.losses)
            self.logger.record("custom/draws", self.draws)
            print(f"[Step {self.num_timesteps}] Win rate: {win_rate:.2f}")

        return True


# ✅ Check CUDA availability
cuda_available = is_cuda_available()
register_chess_env()

# ✅ Create environment
def make_env():
    env = gym.make("gymnasium_env/ChessGame-v0")
    return Monitor(env)

if __name__ == "__main__":  # ✅ Required for Windows
    # ✅ Create SubprocVecEnv for parallel environments
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        print("Training PPO agent with GPU and parallel environments...")
        model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,        # Could later tune to 1e-5 if overfitting
        n_steps=N_STEPS,              # Rollout steps per environment
        batch_size=BATCH_SIZE,            # Ensure divisibility (n_steps * n_envs) % batch_size == 0
        n_epochs=N_EPOCHS,               # PPO update passes
        gamma=0.3,                 # Low discount factor as per research
        gae_lambda=1.0,            # Full advantage estimation
        clip_range=0.2,
        ent_coef=0.01,             # Encourages exploration
        vf_coef=0.5,               # Value loss coefficient (default)
        max_grad_norm=0.5,
        device="cuda" if cuda_available else "cpu",
        tensorboard_log=LOG_DIR)
        
        summary(model.policy)

        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="PPO_Chess",callback=callback)
        model.save(MODEL_PATH)
        print(f"✅ Model saved as {MODEL_PATH}")
        del model

        # ✅ Load model and evaluate
        model = PPO.load(MODEL_PATH, env=env)
        print("Evaluating PPO agent...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# tensorboard --logdir=./chess_logs 