import os
import torch
import torch.nn as nn
import torch.optim as optim
# from stable_baselines3.common.logger import logger
from ChessGame.ChessEnv import register_chess_env
from ChessMaskablePPO import make_env_masking_enabled
from ChessPPO import WinRateCallback
from cudaCheck import is_cuda_available
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO
from torchinfo import summary

# ─── Hyperparameters ─────────────────────────────
MODEL_PATH      = "ppo_AE_chess.zip"
LOG_DIR         = "./chess_logs"
TOTAL_TIMESTEPS = 100_000
N_ENVS          = 10
N_STEPS         = 2048
BATCH_SIZE      = 512
N_EPOCHS        = 10
AE_LR           = 1e-4
# Choose sizes here:
INPUT_DIM  = 5 * 5         # flatten 5×5 board → 25
HIDDEN_DIM = 64            # intermediate layer size
LATENT_DIM = 8             # bottleneck size (e.g. 25/8 ≃3× compression)
# ──────────────────────────────────────────────────

cuda_available = is_cuda_available()
register_chess_env()

from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class AELogCallback(BaseCallback):
    def __init__(self, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
       
    """
    Every rollout step, read the last autoencoder loss from
    policy.features_extractor.last_ae_loss and log it.
    """
    def _on_step(self) -> bool:

        if self.num_timesteps % self.log_interval == 0:
            # Grab the feature extractor
            ae_ext = self.model.policy.features_extractor
            # If it has a valid loss recorded, log it
            if getattr(ae_ext, "last_ae_loss", None) is not None:
                # 'self.logger' is the SB3 logger attached to the model
                self.logger.record("train/ae_loss", ae_ext.last_ae_loss)
        return True


class BoardAutoencoder(nn.Module):
    def __init__(self,
                 input_dim: int = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 latent_dim: int = LATENT_DIM):
        """
        A parameterizable 2-layer autoencoder:
          encoder: input_dim → hidden_dim → latent_dim
          decoder: latent_dim → hidden_dim → input_dim
        """
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid(),              # assume normalized board ∈ [0,1]
            nn.Unflatten(1, (5, 5)),   # rebuild 5×5
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 5, 5)
        returns: (z, recon) where
          z     : (batch, latent_dim)
          recon : (batch, 5, 5)
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

class AutoencoderFeatureExtractor(BaseFeaturesExtractor):
    MIN_VAL, MAX_VAL = -60000.0, 60000.0
    range_val = MAX_VAL - MIN_VAL  # = 120000

    def __init__(self, observation_space, encoder_model: BoardAutoencoder):
        latent_dim = encoder_model.encoder[-1].out_features
        super().__init__(observation_space, features_dim=latent_dim)

        self.device = torch.device("cuda" if cuda_available else "cpu")
        self.autoencoder        = encoder_model.to(self.device)
        self.reconstruction_loss = nn.MSELoss()
        self.optimizer          = optim.Adam(self.autoencoder.parameters(), lr=AE_LR)
        self.last_ae_loss       = None

    def forward(self, observations: dict) -> torch.Tensor:
        board = observations["board"].float().to(next(self.autoencoder.parameters()).device)
        # board = observations["board"].float().to(self.device)
        board_norm = (board - self.MIN_VAL) /  self.range_val   # now in [0,1]
        # ——— Your AE update (force gradients on) ———
        with torch.enable_grad():
            z, recon = self.autoencoder(board_norm)
            loss = self.reconstruction_loss(recon, board_norm)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_ae_loss = loss.item()
        
            # print(f"train/ae_loss: {loss.item()}")

        # Detach z so PPO only trains its own nets
        return z.detach()


if __name__ == "__main__":
    env = make_vec_env(
        make_env_masking_enabled,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        # instantiate AE with whatever dims you like
        ae = BoardAutoencoder(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM
        )

        policy_kwargs = dict(
            features_extractor_class=AutoencoderFeatureExtractor,
            features_extractor_kwargs={"encoder_model": ae},
        )

        print("Training PPO + AE...")
        model = MaskablePPO(
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
            tensorboard_log=LOG_DIR,
        )

        summary(model.policy)

        # callback = WinRateCallback(log_interval=5_000)
        callback = CallbackList([
            WinRateCallback(log_interval=5_000),
            AELogCallback(log_interval=5_000),
        ])

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name="PPO_AE_Chess",
            callback=callback,
        )

        model.save(MODEL_PATH)
        print(f"✅ Model saved as {MODEL_PATH}")
        del model

        # Evaluate
        model = MaskablePPO.load(MODEL_PATH, env=env)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

# To view AE loss alongside PPO metrics:
# tensorboard --logdir=./chess_logs
