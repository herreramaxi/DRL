import os
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from ChessPPO import WinRateCallback
from common import is_cuda_available, make_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchinfo import summary
from transformers import GPT2Config, GPT2Model

# ✅ Hyperparameters
MODEL_PATH = "ppo_transformer_chess.zip"
LOG_DIR = "./chess_logs"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 10
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 10

# ✅ CUDA check
cuda_available = is_cuda_available()
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
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):       
        tiny_transformer = TinyGPT2Encoder(vocab_size=100, seq_len=25, n_layer=2, n_head=2, n_embd=64)

        policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs={"transformer_model": tiny_transformer}
        )

        print("Training PPO-Tranformer agent with GPU and parallel environments...")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=0.3,
            gae_lambda=1.0,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cuda" if cuda_available else "cpu",
            policy_kwargs=policy_kwargs,
            tensorboard_log=LOG_DIR
        )

        summary(model.policy)

        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="PPO_Transformer_Chess", callback=callback)
        model.save(MODEL_PATH)
        print(f"✅ Model saved as {MODEL_PATH}")
        del model

        # ✅ Load and evaluate
        model = PPO.load(MODEL_PATH, env=env)
        print("Evaluating PPO_Transformer_Chess agent...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# ➤ Run TensorBoard with:
# tensorboard --logdir=./chess_logs
