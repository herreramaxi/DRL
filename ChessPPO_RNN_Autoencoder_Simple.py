import os
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from ChessMaskablePPO import make_env_masking_enabled
from ChessPPO import WinRateCallback
from ChessPPO_AutoEncoder import FrozenAEFeatureExtractor
from ae_pretrain import BoardAutoencoder
from cudaCheck import is_cuda_available
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskableRecurrentPPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from torchinfo import summary

# ─── Hyperparameters ─────────────────────────────
MODEL_PATH      = "ppo_AE_RNN_simple_chess.zip"
LOG_DIR         = "./chess_logs"
TOTAL_TIMESTEPS = 1_000
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

    model = MaskableRecurrentPPO(
        policy="MultiInputLstmPolicy",
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
                tb_log_name="PPO_AE_RNN_Simple_Chess",
                callback=callback)

    model.save(MODEL_PATH)
    print(f"✅ PPO_AE_RNN simple model saved as {MODEL_PATH}")

    # Evaluate
    model = MaskableRecurrentPPO.load(MODEL_PATH, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
