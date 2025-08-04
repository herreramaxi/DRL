import os
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from Chess_1_PPO import WinRateCallback
from common import get_device_name, is_cuda_available, make_env, make_env_masking_enabled, parse_arguments, print_model_summary
from custom_logging import important, success
import gymnasium as gym
# from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO    
# from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchinfo import summary
from transformers import GPT2Config, GPT2Model

register_chess_env()

class TinyGPT2Encoder(nn.Module):
    def __init__(self, vocab_size=100, seq_len=25, n_layer=2, n_head=2, n_embd=64):
        super().__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_len,
            n_ctx=seq_len,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.transformer = GPT2Model(self.config)

    def forward(self, x):
        return self.transformer(inputs_embeds=x).last_hidden_state[:, -1, :]

# class TransformerFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, transformer_model):
#         super().__init__(observation_space, features_dim=transformer_model.config.n_embd)
#         self.transformer = transformer_model

#     def forward(self, observations):
#         board = observations["board"].view(observations["board"].size(0), -1)  # e.g. (batch, 25)
#         x = board.unsqueeze(1).float()  # reshape to (batch, seq_len, emb_dim=1)
#         x = x.repeat(1, 1, self.transformer.config.n_embd)  # make it into fake embeddings
#         return self.transformer(x)

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, transformer_model):
        super().__init__(observation_space, features_dim=transformer_model.config.n_embd)
        self.transformer = transformer_model
        self.project = nn.Linear(25, transformer_model.config.n_embd)  # 25 → 64

    def forward(self, observations):
        board = observations["board"].view(observations["board"].size(0), -1)  # (batch, 25)
        x = self.project(board).unsqueeze(1)  # (batch, 1, n_embd)
        return self.transformer(x)


if __name__ == "__main__":
    args = parse_arguments("8_Naive_Transformer_PPO")   
    device = get_device_name()
    register_chess_env()  

    if not os.path.exists(args.model_path):
        print(f"Training '{args.agent_name}' agent using device '{device}' and '{args.n_envs}' parallel environments...")
        
        env = make_vec_env(make_env_masking_enabled, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

        tiny_transformer = TinyGPT2Encoder(vocab_size=100, seq_len=25, n_layer=2, n_head=2, n_embd=64)
        policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs={"transformer_model": tiny_transformer}
        )

        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.log_dir
        )

        print_model_summary(model)

        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.agent_name,callback=callback)
        model.save(args.model_path)
        success(f"Model saved on {args.model_path}")
        del model

    env = make_vec_env(make_env_masking_enabled, n_envs=1, vec_env_cls=DummyVecEnv)
    model = MaskablePPO.load(args.model_path, env=env)
    print(f"Evaluating {args.agent_name} agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    important(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
# ➤ Run TensorBoard with:
# tensorboard --logdir=./chess_logs
