import os
import argparse
import numpy as np
from ChessGame.ChessEnv import register_chess_env
from ChessPPO import WinRateCallback
from common import get_device_name, get_model_path, make_env_masking_enabled, parse_arguments
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.ppo_mask_recurrent.ppo_mask_recurrent import MaskableRecurrentPPO
from torchinfo import summary

if __name__ == "__main__":  
    args = parse_arguments("MaskableRecurrentPPO")   
    device = get_device_name()
    register_chess_env()  
    
    if not os.path.exists(args.model_path):
        print(f"Training '{args.agent_name}' agent using device '{device}' and '{args.n_envs}' parallel environments...")
        
        env = make_vec_env(make_env_masking_enabled, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

        model = MaskableRecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=env,
            verbose=1,
            learning_rate=1e-4,        # Could later tune to 1e-5 if overfitting
            n_steps=args.n_steps,              # Rollout steps per environment
            batch_size=args.batch_size,            # Ensure divisibility (n_steps * n_envs) % batch_size == 0
            n_epochs=args.n_epochs,               # PPO update passes
            gamma=0.3,                 # Low discount factor as per research
            gae_lambda=1.0,            # Full advantage estimation
            clip_range=0.2,
            ent_coef=0.01,             # Encourages exploration
            vf_coef=0.5,               # Value loss coefficient (default)
            max_grad_norm=0.5,
            device=device,
            tensorboard_log=args.log_dir)

        print("Model summary:")    
        summary(model.policy)
        
        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.agent_name,callback=callback)
        model.save(args.model_path)
        print(f"Model saved on {args.model_path}")
        del model

    env = make_vec_env(make_env_masking_enabled, n_envs=1, vec_env_cls=DummyVecEnv)
    model = MaskableRecurrentPPO.load(args.model_path, env=env)
    print(f"Evaluating {args.agent_name} agent...")
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