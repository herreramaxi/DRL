import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from GridWorld.GridWorldEnv import register_gridworld_env
from GridWorld.GridWorldStateWrapper import GridWorldStateWrapper

MODEL_PATH = "ppo_gridworld.zip"
LOG_DIR = "./ppo_logs"
TOTAL_TIMESTEPS = 100_000
N_ENVS = 16

register_gridworld_env()

def make_env():
    env = gym.make("gymnasium_env/GridWorld-v0")
    env = GridWorldStateWrapper(env)
    return Monitor(env)

if __name__ == "__main__":  # ✅ IMPORTANT for Windows multiprocessing
    # Create SubprocVecEnv for parallel training
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        print("Training PPO agent with GPU and parallel environments...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,       # Lower for faster updates
            batch_size=1024,    # Large batch for efficiency
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            # device="cuda",      # Use GPU
            device="cpu",
            tensorboard_log=LOG_DIR
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="PPO_GridWorld")
        model.save(MODEL_PATH)
        print(f"Model saved as {MODEL_PATH}")
        del model

    model = PPO.load(MODEL_PATH, env=env)

    print("Evaluating PPO agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
