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

        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
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
        from torchview import draw_graph

        # build a dummy obs matching your Dict space
        obs_space = model.observation_space
        dummy_obs = {
            "actions": obs_space["actions"].sample(),
            "board":   obs_space["board"].sample(),
        }

        # convert to tensors on the right device
        obs_tensors, _ = model.policy.obs_to_tensor(dummy_obs)  # -> dict of tensors

        # IMPORTANT: pass as a single positional argument
        graph = draw_graph(
            model.policy,
            input_data=(obs_tensors,),     # tuple! prevents kwargs expansion
            expand_nested=True
        )

        graph.visual_graph.render("ppo_architecture", format="png")
        print("Saved: ppo_architecture.png")
 
        # obs_space = model.observation_space  # this is the single-env space even for VecEnvs
        # dummy_obs = {
        #     "actions": obs_space["actions"].sample(),  # shape (942,)
        #     "board":   obs_space["board"].sample(),    # shape (5, 5)
        # }

        # # 2) Convert to tensor on the correct device (adds batch dim internally)
        # obs_tensor, _ = model.policy.obs_to_tensor(dummy_obs)

        # # 3) Get actor distribution (for logits) and value
        # dist = model.policy.get_distribution(obs_tensor)  # Categorical for Discrete(942)
        # logits = dist.distribution.logits                 # shape: (batch, 942)

        # # Value path (shape: (batch, 1))
        # values = model.policy.predict_values(obs_tensor)

        # # 4) Combine to a scalar so torchviz traces both branches in one graph
        # combined = logits.sum() + values.sum()

        # # 5) Draw and render
        # from torchviz import make_dot
        # graph = make_dot(
        #     combined,
        #     params=dict(model.policy.named_parameters())
        # )
        # graph.render("Chess_1_PPO", format="png", cleanup=True)
        # print("Saved: Chess_1_PPO.png")

    evaluate_model(PPO, args, evaluate_policy) 

# tensorboard --logdir=./chess_logs 