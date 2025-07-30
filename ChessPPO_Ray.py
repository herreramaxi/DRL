import os
import gymnasium as gym
from ChessGame.ChessEnv import register_chess_env
from cudaCheck import is_cuda_available
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
import ray

# ✅ Hyperparameters
MODEL_PATH = "rllib_ppo_chess"
LOG_DIR = "./chess_logs"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 10
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 10

# ✅ Check CUDA availability
cuda_available = is_cuda_available()
print(f"cuda_avialable: {cuda_available}")

# ✅ Register chess environment
register_chess_env()


# ✅ Create environment and register with Ray
def make_env(env_config=None):
    env = gym.make("gymnasium_env/ChessGame-v0")
    return env

register_env("ChessGame-v0", make_env)


# ✅ RLlib-compatible win-rate callback
class WinRateCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for()
        if info and "result" in info:
            result = info["result"]
            episode.custom_metrics["win"] = int(result == "win")
            episode.custom_metrics["loss"] = int(result == "loss")
            episode.custom_metrics["draw"] = int(result == "draw")
        if info and "move" in info:
            episode.custom_metrics["invalid_moves"] = int(info["move"] == "invalid")
            episode.custom_metrics["valid_moves"] = int(info["move"] != "invalid")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment(env="ChessGame-v0")
        .framework("torch")
        .resources(num_gpus=1 if cuda_available else 0)
        .env_runners(num_env_runners=N_ENVS)
        .training(
            gamma=0.3,
            lambda_=1.0,
            lr=1e-4,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=0.5
        )
        .callbacks(WinRateCallback)
        # .logger_config({"logdir": LOG_DIR})
    )

    config.logger_config = {"logdir": LOG_DIR}
    algo = config.build()

    print("Training PPO agent with RLlib...")
    steps = 0
    while steps < TOTAL_TIMESTEPS:
        result = algo.train()
        steps = result["timesteps_total"]
        win_rate = result["custom_metrics"].get("win_mean", 0.0)
        print(f"[Step {steps}] Win rate: {win_rate:.2f}")

    checkpoint = algo.save(MODEL_PATH)
    print(f"✅ Model saved at: {checkpoint}")

    ray.shutdown()
