import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from GridWorld.GridWorldEnv import register_gridworld_env
from GridWorld.GridWorldStateWrapper import GridWorldStateWrapper

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should

# ==========================
# Configuration
# ==========================
MODEL_PATH = "dqn_gridworld.zip"
LOG_DIR = "./dqn_logs"
TOTAL_TIMESTEPS = 20_000  # Increased for better learning

# Register custom environment
register_gridworld_env()

def make_env():
    env = gym.make("gymnasium_env/GridWorld-v0")
    env = GridWorldStateWrapper(env)
    return Monitor(env)

if __name__ == "__main__":
    env = make_env()  # ✅ Single environment (DQN does not use VecEnv)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    if not os.path.exists(MODEL_PATH):
        print("Training DQN agent with GPU...")

        model = DQN(
            policy="MlpPolicy",
            env=env,
            # learning_rate=1e-3,        # Faster learning
            buffer_size=100_000,        # Replay buffer size
            # learning_starts=1_000,     # Warm-up steps
            batch_size=2048,             # Batch size for gradient updates
            # gamma=0.99,                # Discount factor
            # train_freq=4,              # Update every 4 steps
            # gradient_steps=1,          # Gradient steps per update
            # exploration_fraction=0.3,  # Explore 30% of training
            # exploration_final_eps=0.05,# Min epsilon
            # target_update_interval=500,# Update target net every 500 steps
            device="cuda",             # ✅ Use GPU
            verbose=1,
            tensorboard_log=LOG_DIR
        )

        # Train
        model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10)
        model.save(MODEL_PATH)
        print(f"Model saved as {MODEL_PATH}")
        del model

    # Load model
    model = DQN.load(MODEL_PATH, env=env)

    print("Evaluating DQN agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # # Test run
    # obs, _ = env.reset()
    # done = False
    # while not done:
    #     action, _ = model.predict(obs)
    #     obs, reward, terminated, truncated, _ = env.step(action)
    #     done = terminated or truncated
    #     print(f"Action: {action}, Reward: {reward}, State: {obs}")


# Use a single environment for testing (not vectorized)
test_env = make_env()
obs, _ = test_env.reset()
done = False
print("\nRunning one test episode:")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)  # Convert from numpy array to int
    obs, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    print(f"Action: {action}, Reward: {reward}, State: {obs}")

print("Test episode complete.")
