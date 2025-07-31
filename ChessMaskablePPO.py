import os
from ChessGame.ChessEnv import register_chess_env
from ChessPPO import WinRateCallback
from cudaCheck import is_cuda_available
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torchinfo import summary

from sb3_contrib import MaskablePPO              
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

# ✅ Hyperparameters
MODEL_PATH = "maskablePPO_chess.zip"
LOG_DIR = "./chess_logs"
TOTAL_TIMESTEPS = 1_000_000  # ✅ Increased for meaningful training
N_ENVS = 10  # ✅ Parallel envs for speed
N_STEPS = 2048  # ✅ More stable with PPO
BATCH_SIZE = 512  # ✅ Must divide n_steps * n_envs (2048 * 8 = 16384)
N_EPOCHS = 10  # ✅ PPO update passes


cuda_available = is_cuda_available()
register_chess_env()

# ✅ Create environment
def make_env_masking_enabled():
    env =  gym.make("gymnasium_env/ChessGame-v0",invalid_action_masking=True)
    return Monitor(env)

if __name__ == "__main__":  # ✅ Required for Windows
    # ✅ Create SubprocVecEnv for parallel environments
    env = make_vec_env(make_env_masking_enabled, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        print("Training PPO agent with GPU and parallel environments...")
        model = MaskablePPO(                     
        # policy="MultiInputPolicy",              
        policy="MlpPolicy", 
        env=env,                              
        learning_rate=1e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.3,
        gae_lambda=1.0,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda" if cuda_available else "cpu",
        tensorboard_log=LOG_DIR,
        verbose=1,
        )
                
        summary(model.policy)

        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="MaskablePPO_Chess",callback=callback)
        model.save(MODEL_PATH)
        print(f"✅ Model saved as {MODEL_PATH}")
        del model

    # ✅ Load model and evaluate
    model = MaskablePPO.load(MODEL_PATH, env=env)
    print("Evaluating MaskablePPO agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # ---------- manual play loop ----------
    # obs = env.reset()
    # while True:
    #     unwrapped= env.unwrapped
    #     action_masks = get_action_masks(env) 
    #     action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
    #     obs, reward, terminated, truncated, info = unwrapped.step(action) # return obs, reward, done, truncated, info    
    #     # print(obs, reward, terminated, truncated )
    #     if terminated.any() or truncated.any():
    #         break
# tensorboard --logdir=./chess_logs 