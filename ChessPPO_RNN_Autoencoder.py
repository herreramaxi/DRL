import os
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from ChessMaskablePPO import make_env_masking_enabled
from ChessPPO import WinRateCallback
from cudaCheck import is_cuda_available
import gymnasium as gym
from sb3_contrib import MaskablePPO    
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchinfo import summary

# ✅ Hyperparameters
MODEL_PATH = "ppo_RNN_AE_chess.zip"
LOG_DIR = "./chess_logs"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 10
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 10

# ✅ CUDA check
cuda_available = is_cuda_available()
register_chess_env()

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                                    num_layers=num_layers, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=input_dim,
                                    num_layers=num_layers, batch_first=True)

    def encode(self, x):  # x: (batch, seq_len=5, input_dim=5)
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.hidden_to_latent(h_n[-1])
        return latent

    def decode(self, z):  # z: (batch, latent_dim)
        h_dec = self.latent_to_hidden(z).unsqueeze(0)  # (1, batch, hidden_dim)
        c_dec = torch.zeros_like(h_dec)
        dec_input = h_dec.transpose(0, 1).repeat(1, 5, 1)  # initial seq: (batch, 5, hidden_dim)
        output, _ = self.decoder_lstm(dec_input, (h_dec, c_dec))
        return output

    def forward(self, x):
        return self.encode(x)

    def reconstruct(self, z):
        return self.decode(z)

class LSTMAutoencoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, encoder_model):
        super().__init__(observation_space, features_dim=encoder_model.latent_dim)
        self.encoder_model = encoder_model

    def forward(self, observations):
        board = observations["board"].float()  # (batch, 5, 5)
        return self.encoder_model.encode(board)

if __name__ == "__main__":
    env = make_vec_env(make_env_masking_enabled, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        # ✅ Instantiate autoencoder
        board_autoencoder = LSTMAutoencoder(input_dim=5, hidden_dim=64, latent_dim=32)

        # ✅ Define policy_kwargs
        policy_kwargs = dict(
        features_extractor_class=LSTMAutoencoderFeatureExtractor,
        features_extractor_kwargs={"encoder_model": board_autoencoder}
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
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="PPO_RNN_AE_Chess", callback=callback)
        model.save(MODEL_PATH)
        print(f"✅ Model saved as {MODEL_PATH}")
        del model

        # ✅ Load and evaluate
        model = MaskablePPO.load(MODEL_PATH, env=env)
        print("Evaluating PPO RNN_AE agent...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# ➤ Run TensorBoard with:
# tensorboard --logdir=./chess_logs
