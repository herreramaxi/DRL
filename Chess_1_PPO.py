import os
from ChessGame.ChessEnv import register_chess_env
from common import get_device_name, parse_arguments, print_model_summary
from commonCallbacks import WinRateCallback
from custom_logging import important, success
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
         
        model = PPO(
            policy="MultiInputPolicy",
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

        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.agent_name,callback=callback)
        model.save(args.model_path)
        success(f"Model saved on {args.model_path}")
        del model

    env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
    model = PPO.load(args.model_path, env=env)
    print(f"Evaluating {args.agent_name} agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    important(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# tensorboard --logdir=./chess_logs 