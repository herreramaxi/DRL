import argparse
import os
import torch
from torchinfo import summary
from datetime import datetime
from commonCallbacks import WinRateCallback
from custom_logging import important, important2, info, success
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

def is_cuda_available():    
    return torch.cuda.is_available()

def get_device_name():
    if torch.cuda.is_available(): 
          return "cuda"
    
    return "cpu"

def make_env_masking_enabled(no_wrapper = False):
    env =  gym.make("gymnasium_env/ChessGame-v0",invalid_action_masking=True, original_step=False) 
        
    if no_wrapper:
        return env
    
    return Monitor(env)
 
def make_env(no_wrapper = False):
    env = gym.make("gymnasium_env/ChessGame-v0",invalid_action_masking=False, original_step=False) 

    if no_wrapper:
        return env
    
    return Monitor(env)

def get_model_path(parent_folder, name,extension=".zip"):  
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{stamp}{extension}"
    model_path = os.path.join(parent_folder, filename)
    return model_path

def parse_arguments(agent_name, total_timesteps= 1_000_000, n_envs=4, n_steps=512, batch_size=256, n_epochs=10, n_samples_ae = 1_000_000 ,n_epochs_ae=20, force_clean = "False", share_features_extractor = False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name",   type=str,   default=agent_name)
    parser.add_argument("--model-path",   type=str,   default=None)
    parser.add_argument("--log-dir",      type=str,   default="./chess_logs")
    parser.add_argument("--total-timesteps", type=int, default=total_timesteps)
    parser.add_argument("--n-envs",       type=int,   default=n_envs) 
    parser.add_argument("--n-steps",      type=int,   default=n_steps)
    parser.add_argument("--batch-size",   type=int,   default=batch_size)
    parser.add_argument("--n-epochs",     type=int,   default=n_epochs)     
    parser.add_argument("--n-samples-ae",     type=int,   default=n_samples_ae)    
    parser.add_argument("--n-epochs-ae",     type=int,   default=n_epochs_ae)    
    parser.add_argument("--force-clean-ae",     type=str,   default=force_clean)    
    parser.add_argument( "--share-features-extractor",action="store_true",help="Share the features extractor between actor and critic")    
    parser.add_argument( "--print-model-only",action="store_true",help="Print model summary only, it does not train or evaluate model")    

    args = parser.parse_args()

    args.model_path = args.model_path if args.model_path else get_model_path("models",args.agent_name)
    
    info("Parsed arguments:")
    for arg, value in vars(args).items():
        info(f"   {arg}: {value}")

    return args

def parse_arguments_ae(n_samples= 1_000_000, board_file_path=None, weights_file_path=None, n_epochs=20, force_clean = "False"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=n_samples)
    parser.add_argument("--board-file-path", type=str, default=board_file_path)
    parser.add_argument("--weights-file-path", type=str, default=weights_file_path) 
    parser.add_argument("--n-epochs", type=int,   default=n_epochs)  
    parser.add_argument("--force-clean", type=str,   default=force_clean)  
    
    args = parser.parse_args()
    
    info("Parsed arguments:")
    for arg, value in list(vars(args).items()):   
        info(f"   {arg}: {value}")
        # arg_nosuffix = arg.replace("_ae","")
        # setattr(args, arg_nosuffix, value)

    return args

def build_cmd(script, args):
    return ["python", script] + args

def print_model_summary(model):
    important("Model summary:")    
    summary(model.policy)

def model_learn(args, model):
    if(args.print_model_only):
        important2("Skipping model.learn(), print_model_only is True")
        return
    
    callback = WinRateCallback(log_interval=5000)
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.agent_name,callback=callback)
    model.save(args.model_path)
    success(f"Model saved on {args.model_path}")
    del model
    
def evaluate_model(model_class, args, evaluate_policy):
    if(args.print_model_only):
        important2("Skipping evaluate_model(), print_model_only is True")
        return
    
    env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
    model = model_class.load(args.model_path, env=env)
    print(f"Evaluating {args.agent_name} agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    important(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    
def evaluate_model_masking_enabled(model_class, args, evaluate_policy):
    if(args.print_model_only):
        important2("Skipping evaluate_model_masking_enabled(), print_model_only is True")
        return
    
    env = make_vec_env(make_env_masking_enabled, n_envs=1, vec_env_cls=DummyVecEnv)
    model = model_class.load(args.model_path, env=env)
    print(f"Evaluating {args.agent_name} agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    important(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")