import os
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from ChessMaskablePPO import make_env_masking_enabled
from ChessPPO import WinRateCallback
from cudaCheck import is_cuda_available
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO     
from torchinfo import summary

# ✅ Hyperparameters
MODEL_PATH = "ppo_AE_chess.zip"
LOG_DIR = "./chess_logs"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 10
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 10

# ✅ CUDA check
cuda_available = is_cuda_available()
register_chess_env()

# ✅ Autoencoder definition
class BoardAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 25),
            nn.Unflatten(1, (5, 5))
        )

    def forward(self, x):
        return self.encoder(x)

    def reconstruct(self, z):
        return self.decoder(z)

# ✅ Feature extractor wrapper
class AutoencoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, encoder_model):
        super().__init__(observation_space, features_dim=encoder_model.encoder[-1].out_features)
        self.encoder = encoder_model.encoder

    def forward(self, observations):
        board = observations["board"]  # (batch, 5, 5)
        x = board.view(board.size(0), -1)  # (batch, 25)
        return self.encoder(x)

if __name__ == "__main__":
    env = make_vec_env(make_env_masking_enabled, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        # ✅ Instantiate autoencoder
        board_autoencoder  = BoardAutoencoder(latent_dim=64)

        # ✅ Define policy_kwargs
        policy_kwargs = dict(
            features_extractor_class=AutoencoderFeatureExtractor,
            features_extractor_kwargs={"encoder_model": board_autoencoder }
        )

        print("Training PPO-AE agent with GPU and parallel environments...")
        model = MaskablePPO(
            policy="MlpPolicy",
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
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="PPO_AE_Chess", callback=callback)
        model.save(MODEL_PATH)
        print(f"✅ Model saved as {MODEL_PATH}")
        del model

        # ✅ Load and evaluate
        model = MaskablePPO.load(MODEL_PATH, env=env)
        print("Evaluating PPO-AE agent...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# ➤ Run TensorBoard with:
# tensorboard --logdir=./chess_logs
