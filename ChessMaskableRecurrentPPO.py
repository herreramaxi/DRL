import os

import numpy as np
from ChessGame.ChessEnv import register_chess_env
from ChessPPO import WinRateCallback
from cudaCheck import is_cuda_available
import gymnasium as gym
# from sb3_contrib import RecurrentPPO
from sb3_contrib import MaskableRecurrentPPO
# from common.ppo_mask_recurrent import RecurrentMaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from torchinfo import summary

# ✅ Hyperparameters
MODEL_PATH = "ppo_recurrent_maskable_chess.zip"
LOG_DIR = "./chess_logs"
TOTAL_TIMESTEPS = 500  # ✅ Increased for meaningful training
N_ENVS = 10  # ✅ Parallel envs for speed
N_STEPS = 2048  # ✅ More stable with PPO
BATCH_SIZE = 512  # ✅ Must divide n_steps * n_envs (2048 * 8 = 16384)
N_EPOCHS = 10  # ✅ PPO update passes

# # ✅ Check CUDA availability
cuda_available = is_cuda_available()
register_chess_env()

# # ✅ Create environment
# def make_env():
#     env = gym.make("gymnasium_env/ChessGame-v0")
#     env = ChessObservationFlattenWrapper(env)
#     return Monitor(env)
def make_env():
    env = gym.make("gymnasium_env/ChessGame-v0", invalid_action_masking=True)
    return Monitor(env)

if __name__ == "__main__":  # ✅ Required for Windows
    # ✅ Create SubprocVecEnv for parallel environments
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        print("Training RecurrentMaskablePPO agent with GPU and parallel environments...")
        model = MaskableRecurrentPPO(
        policy="MultiInputLstmPolicy",
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
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="RecurrentMaskablePPO_Chess",callback=callback)
        model.save(MODEL_PATH)
        print(f"✅ Model saved as {MODEL_PATH}")
        del model

        # ✅ Load model and evaluate
        model = MaskableRecurrentPPO.load(MODEL_PATH, env=env)
        print("Evaluating RecurrentMaskablePPO agent...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# # tensorboard --logdir=./chess_logs 

# if __name__ == "__main__": 
#     env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
#     print("env.observation_space")
#     print(env.observation_space)

#     print("env.observation_space.spaces['board']")
#     print(env.observation_space.spaces['board'])
#     print("env.observation_space.spaces['board'].shape")
#     print(env.observation_space.spaces['board'].shape)

#     print("env.observation_space.spaces['actions']")
#     print(env.observation_space.spaces["actions"])
#     print("env.observation_space.spaces['actions'].shape")
#     print(env.observation_space.spaces["actions"].shape)
    # #         actions_shape = env.observation_space.spaces["actions"].shape  # (N_ACTIONS,)