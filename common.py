import argparse
import os
import torch
from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

def is_cuda_available():    
    return torch.cuda.is_available()

def get_device_name():
    if torch.cuda.is_available(): 
          return "cuda"
    
    return "cpu"

def make_env_masking_enabled():
    env =  gym.make("gymnasium_env/ChessGame-v0",invalid_action_masking=True, original_step=False) 
    return Monitor(env)
 
def make_env():
    env = gym.make("gymnasium_env/ChessGame-v0",invalid_action_masking=False, original_step=False) 
    return Monitor(env)

def get_model_path(parent_folder, name,extension=".zip"):  
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{stamp}{extension}"
    model_path = os.path.join(parent_folder, filename)
    return model_path

def parse_arguments(agent_name, total_timesteps= 1_000_000, n_envs=10, n_steps=2048, batch_size=512, n_epochs=10):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name",   type=str,   default=agent_name)
    parser.add_argument("--model-path",   type=str,   default=None)
    parser.add_argument("--log-dir",      type=str,   default="./chess_logs")
    parser.add_argument("--total-timesteps", type=int, default=total_timesteps)
    parser.add_argument("--n-envs",       type=int,   default=n_envs)
    parser.add_argument("--n-steps",      type=int,   default=n_steps)
    parser.add_argument("--batch-size",   type=int,   default=batch_size)
    parser.add_argument("--n-epochs",     type=int,   default=n_epochs)      
    args = parser.parse_args()

    args.model_path = args.model_path if args.model_path else get_model_path("models",args.agent_name)
    
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"   {arg}: {value}")

    return args
