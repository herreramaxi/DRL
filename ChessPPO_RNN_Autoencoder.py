import os
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from ChessMaskablePPO import make_env_masking_enabled
from ChessPPO import WinRateCallback
from ae_rnn_pretrain import SimpleLSTMAutoencoder
from cudaCheck import is_cuda_available
import gymnasium as gym
from sb3_contrib import MaskablePPO   
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchinfo import summary


# ─── Hyperparameters ─────────────────────────────
MODEL_PATH      = "ppo_AE_RNN_chess.zip"
LOG_DIR         = "./chess_logs"
TOTAL_TIMESTEPS = 1_000
N_ENVS          = 10
N_STEPS         = 2048
BATCH_SIZE      = 512
N_EPOCHS        = 10
INPUT_DIM       = 5 * 5
HIDDEN_DIM      = 64
LATENT_DIM      = 8
AE_WEIGHTS_PATH = "ae_rnn_pretrained.pth"
# ──────────────────────────────────────────────────
# ✅ CUDA check
cuda_available = is_cuda_available()
register_chess_env()

class LSTMAutoencoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, encoder_model):
        # latent_dim = encoder_model.encoder_lstm[-1].out_features
        latent_dim = encoder_model.to_latent.out_features
        super().__init__(observation_space, features_dim=latent_dim)
        self.enc = encoder_model
        self.encoder_model = encoder_model
        self.encoder = encoder_model.encoder_lstm

    def forward(self, observations):
        b = observations["board"].float()
        b = (b + 60000.0) / 120000.0
        # return self.encoder(b)
        return self.enc.encode_single(b)

if __name__ == "__main__":
    env = make_vec_env(make_env_masking_enabled, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        device = torch.device("cuda" if cuda_available else "cpu")
        # 1) Load & freeze your pre-trained RNN AE
        ae = SimpleLSTMAutoencoder()
        
        ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
        ae.eval()
        for p in ae.parameters(): p.requires_grad = False

        # ✅ Define policy_kwargs
        policy_kwargs = dict(
        features_extractor_class=LSTMAutoencoderFeatureExtractor,
        features_extractor_kwargs={"encoder_model": ae}
        )

        print("Training PPO-AE-RNN agent with GPU and parallel environments...")
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
