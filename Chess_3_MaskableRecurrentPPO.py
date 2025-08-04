import os
import argparse
import numpy as np
from ChessGame.ChessEnv import register_chess_env
from Chess_1_PPO import WinRateCallback
from common import evaluate_model_masking_enabled, get_device_name, get_model_path, make_env_masking_enabled, model_learn, parse_arguments, print_model_summary
from custom_logging import important, important2, success
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.ppo_mask_recurrent.ppo_mask_recurrent import MaskableRecurrentPPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from torchinfo import summary

register_chess_env()  

if __name__ == "__main__":  
    args = parse_arguments("3_MaskableRecurrentPPO")   
    device = get_device_name()    
    
    if not os.path.exists(args.model_path):
        print(f"Training '{args.agent_name}' agent using device '{device}' and '{args.n_envs}' parallel environments...")
        
        env = make_vec_env(make_env_masking_enabled, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
        important2(f"share_features_extractor: {args.share_features_extractor}")
        
        model = MaskableRecurrentPPO(
            policy="MultiInputLstmPolicy",
            policy_kwargs=dict(share_features_extractor=args.share_features_extractor),
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

        print_model_summary(model)
        model_learn(args, model)

    evaluate_model_masking_enabled(MaskableRecurrentPPO, args, evaluate_policy)        


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