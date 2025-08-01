import os
import torch
import torch.nn as nn
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
INPUT_DIM       = 5 * 5
HIDDEN_DIM      = 64
LATENT_DIM      = 8
AE_WEIGHTS_PATH = "ae_pretrained.pth"
# ──────────────────────────────────────────────────

cuda_available = is_cuda_available()
register_chess_env()

# Same AE class definition
class BoardAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, (5,5)),
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

# Feature extractor uses only the encoder, no training here
class FrozenAEFeatureExtractor(BaseFeaturesExtractor):
    MIN_VAL, MAX_VAL = -60000.0, 60000.0
    def __init__(self, observation_space, encoder_model: BoardAutoencoder):
        latent_dim = encoder_model.encoder[-1].out_features
        super().__init__(observation_space, features_dim=latent_dim)
        self.encoder = encoder_model.encoder
    def forward(self, observations: dict) -> torch.Tensor:
        board = observations["board"].float()
        # normalize to [0,1]
        board = (board - self.MIN_VAL) / (self.MAX_VAL - self.MIN_VAL)
        return self.encoder(board)

if __name__ == "__main__":
    # Load & freeze AE
    device = torch.device("cuda" if cuda_available else "cpu")
    ae = BoardAutoencoder(input_dim=INPUT_DIM,
                          hidden_dim=HIDDEN_DIM,
                          latent_dim=LATENT_DIM).to(device)
    # ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
    state_dict = torch.load(
    AE_WEIGHTS_PATH,
    map_location=device,
    weights_only=True    # ← new argument
    )
    ae.load_state_dict(state_dict)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # Build PPO
    policy_kwargs = dict(
        features_extractor_class=FrozenAEFeatureExtractor,
        features_extractor_kwargs={"encoder_model": ae},
    )
    env = make_vec_env(make_env_masking_enabled,
                       n_envs=N_ENVS,
                       vec_env_cls=SubprocVecEnv)

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
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
    )

    summary(model.policy)

    # Train
    callback = WinRateCallback(log_interval=5_000)
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                tb_log_name="PPO_AE_Chess_Frozen",
                callback=callback)

    model.save(MODEL_PATH)
    print(f"✅ PPO_AE model saved as {MODEL_PATH}")

    # Evaluate
    model = MaskablePPO.load(MODEL_PATH, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
