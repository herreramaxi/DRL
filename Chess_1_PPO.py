import os
from ChessGame.ChessEnv import register_chess_env
from common import evaluate_model, get_device_name, model_learn, parse_arguments, print_model_summary
from commonCallbacks import WinRateCallback
from custom_logging import important, important2, success
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from torchinfo import summary
from common import make_env

register_chess_env()  

if __name__ == "__main__":
    args = parse_arguments("1_PPO")   
    device = get_device_name()    

    if not os.path.exists(args.model_path):
        print(f"Training '{args.agent_name}' agent using device '{device}' and '{args.n_envs}' parallel environments...")        
        env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

        # print("Observation space:", env.observation_space)
        # print("Action space:", env.action_space)
        important2(f"share_features_extractor: {args.share_features_extractor}")

        model = PPO(
            policy="MultiInputPolicy",
            policy_kwargs=dict(share_features_extractor=args.share_features_extractor),
            env=env,
            verbose=1,
            learning_rate=1e-4,        # Could later tune to 1e-5 if overfitting
            n_steps=args.n_steps,              # Rollout steps per environment
            batch_size=args.batch_size,            # Ensure divisibility (n_steps * n_envs) % batch_size == 0
            n_epochs=args.n_epochs,          # PPO update passes
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

    evaluate_model(PPO, args, evaluate_policy) 

# tensorboard --logdir=./chess_logs 