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
import random
from sb3_contrib.common.maskable.utils import get_action_masks
register_chess_env()  

if __name__ == "__main__":
    args = parse_arguments("1_PPO")   
    device = get_device_name()    

    if not os.path.exists(args.model_path):
        print(f"Training '{args.agent_name}' agent using device '{device}' and '{args.n_envs}' parallel environments...")        
        env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        important2(f"share_features_extractor: {args.share_features_extractor}")

        model = PPO(
            policy="MultiInputPolicy",
            policy_kwargs=dict(share_features_extractor=args.share_features_extractor),
            env=env,
            verbose=1,
            learning_rate=1e-4,         # Learning Rate
            n_steps=args.n_steps,       # Rollout steps per environment
            batch_size=args.batch_size, # Minibatch size
            n_epochs=args.n_epochs,     # Number of Epochs
            gamma=0.3,                  # Discount factor
            gae_lambda=1.0,             # Full advantage estimation
            clip_range=0.2,
            ent_coef=0.01,             
            vf_coef=0.5,               
            max_grad_norm=0.5,
            device=device,
            tensorboard_log=args.log_dir)
        
        import torch
        from gymnasium.spaces import Discrete
        # 1) Inspect the policy head module (Linear to action logits)
        print(model.policy.action_net)  # e.g., Linear(in_features=64, out_features=942, bias=True)

        # 2) Sizes
        print("latent_dim_pi:", model.policy.mlp_extractor.latent_dim_pi)
        print("policy head in_features:", model.policy.action_net.in_features)
        print("policy head out_features:", model.policy.action_net.out_features)

        # (optional) Confirm action-space size from the env:
        print("env action space:", env.action_space)
        if isinstance(env.action_space, Discrete):
            print("n_actions (env):", env.action_space.n)

        # 1) Inspect the value head module (Linear to scalar)
        print(model.policy.value_net)  # e.g., Linear(in_features=..., out_features=1, bias=True)

        # 2) Sizes
        print("latent_dim_vf:", model.policy.mlp_extractor.latent_dim_vf)
        print("value head in_features:", model.policy.value_net.in_features)
        print("value head out_features:", model.policy.value_net.out_features)  # should be 1

        # 3) Run a dummy observation through the critic to see shape/dtype/device
        obs_sample = env.observation_space.sample()  # works for Dict obs too
        obs_tensor, _ = model.policy.obs_to_tensor(obs_sample)  # puts on correct device (CPU/CUDA)

        with torch.no_grad():
            v = model.policy.predict_values(obs_tensor)

        print("value: ")
        print(v)
        print("Output shape:", v.shape)   # -> (batch, 1)
        print("Output dtype:", v.dtype)   # usually torch.float32
        print("Output device:", v.device) # cuda:0 if using GPU

            # ---------- manual play loop ----------
        env = gym.make("gymnasium_env/ChessGame-v0", invalid_action_masking=False)
        obs, _ = env.reset() 
        env.render() 
        
        while True:      
            action_arr, _states   = model.predict(obs)
            
            print("action_arr")
            print(action_arr)
            print("_states ")
            print(_states )
            
            # move = random.choice(action_arr)        

            obs, reward, terminated, truncated, info= env.step(action_arr)
            env.render()
            # print(obs, reward, terminated, truncated )
            if terminated or truncated:
                break

        print_model_summary(model)
        model_learn(args, model)       

    evaluate_model(PPO, args, evaluate_policy) 

# tensorboard --logdir=./chess_logs 