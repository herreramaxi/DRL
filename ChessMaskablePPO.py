import os
import argparse
from ChessGame.ChessEnv import register_chess_env
from ChessPPO import WinRateCallback
from common import get_device_name, get_model_path, make_env_masking_enabled, parse_arguments

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torchinfo import summary

from sb3_contrib import MaskablePPO              
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

register_chess_env()  

if __name__ == "__main__":    
    args = parse_arguments("MaskablePPO")   
    device = get_device_name()
    
    if not os.path.exists(args.model_path):
        print(f"Training '{args.agent_name}' agent using device '{device}' and '{args.n_envs}' parallel environments...")
        
        env = make_vec_env(make_env_masking_enabled, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

        model = MaskablePPO(                     
            policy="MultiInputPolicy",              
            # policy="MlpPolicy", 
            env=env,                              
            learning_rate=1e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.3,
            gae_lambda=1.0,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
            tensorboard_log=args.log_dir,
            verbose=1)

        print("Model summary:")    
        summary(model.policy)

        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.agent_name,callback=callback)
        model.save(args.model_path)
        print(f"Model saved on {args.model_path}")
        del model

    env = make_vec_env(make_env_masking_enabled, n_envs=1, vec_env_cls=DummyVecEnv)
    model = MaskablePPO.load(args.model_path, env=env)
    print(f"Evaluating {args.agent_name} agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # # ---------- manual play loop ----------
    # env = gym.make("gymnasium_env/ChessGame-v0", invalid_action_masking=True)
    # obs, _ = env.reset() 
    # env.render() 
    
    # while True:      
    #     action_masks = get_action_masks(env) 
    #     action_arr, _  = model.predict(obs, action_masks=action_masks, deterministic=False)
        
    #     # unpack the single element
    #     action = int(action_arr)   # or action_arr[0]
    #     if action_masks[action] == 0:
    #     # definitely invalid
    #         print("⚠️  Agent picked a masked-out action:", env.unwrapped.game.id_to_action[action])
            
    #     obs, reward, terminated, truncated, info = env.step(action) # return obs, reward, done, truncated, info    
    #     print("move:",  info["chess_move"]) 
    #     env.render() 
    #     # print(obs, reward, terminated, truncated )
    #     if terminated or truncated:
    #         break
# tensorboard --logdir=./chess_logs 
