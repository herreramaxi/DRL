# play_saved_agent.py
import argparse
from common import make_env_masking_enabled
import gymnasium as gym
from sb3_contrib import MaskablePPO,MaskableRecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.maskable.utils import get_action_masks
from ChessGame.ChessEnv import register_chess_env
from stable_baselines3 import PPO

register_chess_env() 

def make_env():
    env =  gym.make("gymnasium_env/ChessGame-v0",invalid_action_masking=True, original_step=False) 
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--rl-algorithm", required=True, help="RL Algoruthm: PPO, MaskablePPO, MaskableRecurrentPPO")
    parser.add_argument("--model-path", required=True, help="Path to your saved .zip model")
    args = parser.parse_args()
  
    env = make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv)

    ##todo: SELECT POLICY DEPENDING ON PARAMETER
    # Load the saved agent. Passing env lets SB3 rebuild policies correctly.
    # model = MaskablePPO.load(args.model_path, env=env)
    model = None

    if args.rl_algorithm == "PPO":
        model = PPO.load(args.model_path, env=env)
    elif args.rl_algorithm == "MaskablePPO":
        model = MaskablePPO.load(args.model_path, env=env)
    elif args.rl_algorithm == "MaskableRecurrentPPO":
        model = MaskableRecurrentPPO.load(args.model_path, env=env)
    else:
        raise ValueError(f"Unsupported RL algorithm: {args.rl_algorithm}")
    
    obs = env.reset()
    done = False
    truncated = False
    ep_reward = 0.0
    step_idx = 0

    env.env_method("render", indices=0) 

    while not done:
        # masks shape: (n_envs, n_actions), dtype=bool
        masks = get_action_masks(env)
        actions, _ = model.predict(obs, deterministic=True, action_masks=masks)

        obs, rewards, dones, infos = env.step(actions)

        # unwrap single-env vectors
        reward = float(rewards[0])
        done   = bool(dones[0])
        info   = infos[0]
        ep_reward += reward
        step_idx += 1

        if "chess_move" in info:
            print()
            print(f"step_number: {step_idx:02d}, move_sequence: {info['chess_move']}, reward: {reward:.2f}")

        # env.envs[0].render()
        env.env_method("render", indices=0) 

    result = info.get("result", "n/a")
    print(f"Game finished --> result: {result}, total_steps: {step_idx}, total_reward:{ep_reward:.2f}")


# python evaluate_agent.py --rl-algorithm MaskablePPO --model-path models/8_Naive_Transformer_PPO_20250806_221310
# python evaluate_agent.py --rl-algorithm MaskableRecurrentPPO --model-path models/5_FF_Autoencoder_MaskableRecurrentPPO_20250806_171012
# python evaluate_agent.py --rl-algorithm MaskableRecurrentPPO --model-path models/5_FF_Autoencoder_MaskableRecurrentPPO_20250805_002954
# python evaluate_agent.py --rl-algorithm MaskableRecurrentPPO --model-path models/7_LSTM_Autoencoder_MaskableRecurrentPPO_20250806_194219
