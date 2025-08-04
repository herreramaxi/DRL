import os
import subprocess
import sys
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from Chess_2_MaskablePPO import make_env_masking_enabled
from Chess_1_PPO import WinRateCallback
from ae_pretrain import BoardAutoencoder
from common import build_cmd, evaluate_model_masking_enabled, get_device_name, is_cuda_available, model_learn, parse_arguments, print_model_summary
from custom_logging import important, important2, success
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from torchinfo import summary

INPUT_DIM       = 5 * 5
HIDDEN_DIM      = 64
LATENT_DIM      = 8
AE_WEIGHTS_PATH = "weights/ae_pretrained.pth"

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

register_chess_env()  

if __name__ == "__main__":
    args = parse_arguments("4_FF_Autoencoder_MaskablePPO")   
    device = get_device_name()
 
    cmd = build_cmd("ae_pretrain.py", 
    [
        "--n-samples", str(args.n_samples_ae), 
        "--n-epochs", str(args.n_epochs_ae),
        "--force-clean", str(args.force_clean_ae),
        "--weights-file-path", AE_WEIGHTS_PATH
    ])

    subprocess.run(cmd, check=True)

    if not os.path.exists(args.model_path):
        # Load & freeze AE
        ae = BoardAutoencoder(input_dim=INPUT_DIM,
                            hidden_dim=HIDDEN_DIM,
                            latent_dim=LATENT_DIM).to(device)
        # ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
        state_dict = torch.load(AE_WEIGHTS_PATH,map_location=device,weights_only=True)
        ae.load_state_dict(state_dict)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad = False

        # Build PPO
        policy_kwargs = dict(
            features_extractor_class=FrozenAEFeatureExtractor,
            features_extractor_kwargs={"encoder_model": ae},
            share_features_extractor=args.share_features_extractor
        )

        env = make_vec_env(make_env_masking_enabled, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
        important2(f"share_features_extractor: {args.share_features_extractor}")
        
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
            tensorboard_log=args.log_dir,
        )

        print_model_summary(model)
        model_learn(args, model)

    evaluate_model_masking_enabled(MaskablePPO, args, evaluate_policy) 
