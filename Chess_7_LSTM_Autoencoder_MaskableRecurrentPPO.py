import os
import subprocess
import torch
import torch.nn as nn
from ChessGame.ChessEnv import register_chess_env
from Chess_2_MaskablePPO import make_env_masking_enabled
from Chess_1_PPO import WinRateCallback
from Chess_6_LSTM_Autoencoder_MaskablePPO import LSTMAutoencoderFeatureExtractor
from ae_rnn_pretrain import SimpleLSTMAutoencoder
from common import build_cmd, get_device_name, is_cuda_available, parse_arguments, print_model_summary
from custom_logging import important, success
import gymnasium as gym
from sb3_contrib.ppo_mask_recurrent.ppo_mask_recurrent import MaskableRecurrentPPO
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

register_chess_env()  

if __name__ == "__main__":
    args = parse_arguments("7_LSTM_Autoencoder_MaskableRecurrentPPO")   
    device = get_device_name()
    
    cmd = build_cmd("ae_rnn_pretrain.py", 
    [
        "--n-samples", str(args.n_samples_ae), 
        "--n-epochs", str(args.n_epochs_ae),
        "--force-clean", str(args.force_clean_ae),
        "--weights-file-path", AE_WEIGHTS_PATH
    ])

    subprocess.run(cmd, check=True)

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
            features_extractor_kwargs={"encoder_model": ae}
        )

        model = MaskableRecurrentPPO(
            policy="MultiInputLstmPolicy",
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

        callback = WinRateCallback(log_interval=5000)
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.agent_name,callback=callback)
        model.save(args.model_path)
        success(f"Model saved on {args.model_path}")
        del model

    env = make_vec_env(make_env_masking_enabled, n_envs=1, vec_env_cls=DummyVecEnv)
    model = MaskableRecurrentPPO.load(args.model_path, env=env)
    print(f"Evaluating {args.agent_name} agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    important(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# ➤ Run TensorBoard with:
# tensorboard --logdir=./chess_logs
