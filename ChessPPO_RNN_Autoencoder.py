import os
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from ChessMaskablePPO import make_env_masking_enabled
from ChessPPO import WinRateCallback
from ae_rnn_pretrain import SimpleLSTMAutoencoder
from common import get_device_name, is_cuda_available, parse_arguments
import gymnasium as gym
from sb3_contrib import MaskablePPO   
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchinfo import summary

INPUT_DIM       = 5 * 5
HIDDEN_DIM      = 64
LATENT_DIM      = 8
AE_WEIGHTS_PATH = "weights/ae_rnn_pretrained.pth"

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
    args = parse_arguments("RNN_Autoencoder_PPO")   
    device = get_device_name()
    register_chess_env()  

    if not os.path.exists(args.model_path):
        print(f"Training '{args.agent_name}' agent using device '{device}' and '{args.n_envs}' parallel environments...")        
        env = make_vec_env(make_env_masking_enabled, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

        # Load & freeze your pre-trained RNN AE
        ae = SimpleLSTMAutoencoder()        
        ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
        ae.eval()
        for p in ae.parameters(): p.requires_grad = False

        policy_kwargs = dict(
            features_extractor_class=LSTMAutoencoderFeatureExtractor,
            features_extractor_kwargs={"lstm_encoder_model": ae}
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
            tensorboard_log=args.log_dir,
        )

        print("Model summary:")    
        summary(model.policy)

        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.agent_name,callback=callback)
        model.save(args.model_path)
        print(f"Model saved on {args.model_path}")
        del model

    env = make_vec_env(make_env_masking_enabled, n_envs=1, vec_env_cls=DummyVecEnv)
    model = MaskablePPO.load(args.model_path, env=env)
    print(f"Evaluating {args.agent_name} agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# ➤ Run TensorBoard with:
# tensorboard --logdir=./chess_logs
