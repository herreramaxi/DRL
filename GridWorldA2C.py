import os
from cudaCheck import is_cuda_available
import gymnasium as gym
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv  # ✅ Safe for Windows
from GridWorld.GridWorldStateWrapper import GridWorldStateWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv

MODEL_PATH = "a2c_gridworld.zip"
LOG_DIR = "./a2c_logs"
TOTAL_TIMESTEPS = 200_000
N_ENVS = 16

def make_env():
    # ✅ Register inside subprocess/thread
    from GridWorld.GridWorldEnv import register_gridworld_env
    register_gridworld_env()
    env = gym.make("gymnasium_env/GridWorld-v0")
    env = GridWorldStateWrapper(env)
    return Monitor(env)

if __name__ == "__main__":
    is_cuda_available()

    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)  # ✅ No subprocess issue

    if not os.path.exists(MODEL_PATH):
        print("Training A2C agent on GPU with parallel envs...")
        model = A2C("MlpPolicy",
                     vec_env, verbose=1, 
                     device="cpu", 
                    # learning_starts=1_000,     # Warm-up steps
                    tensorboard_log=LOG_DIR)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(MODEL_PATH)
        del model

    model = A2C.load(MODEL_PATH, device="cpu", env=vec_env)
    print("Evaluating...")
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
