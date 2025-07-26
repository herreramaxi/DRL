from GridWorld.GridWorldAgent import GridWorldAgent
from GridWorld.GridWorldStateWrapper import GridWorldStateWrapper
from GridWorld.GridWorldEnv import GridWorldEnv
from collections import defaultdict
import gymnasium as gym
import numpy as np

from matplotlib import pyplot as plt

from typing import Optional
import numpy as np

from gymnasium.utils.env_checker import check_env as gym_check_env

def run_env_check(env):
    try:
        gym_check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

def simpleTest():
    obs, info = env.reset(seed=42)  # Use seed for reproducible testing

    # print(f"Starting position - Agent: {obs['agent']}, Target: {obs['target']}")
    # print(f"Starting state: {obs}")

    # Test each action type
    actions = [0, 1, 2, 3]  # right, up, left, down
    for action in actions:
        old_pos = obs['agent'].copy()
        obs, reward, terminated, truncated, info = env.step(action)
        new_pos = obs['agent']
        print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")


from GridWorld.GridWorldEnv import register_gridworld_env
register_gridworld_env()

# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 5_000        # Number of hands to practice
# n_episodes = 100        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration
# Create environment and agent
env = gym.make("gymnasium_env/GridWorld-v0")
run_env_check(env)  # Check environment validity
simpleTest()  # Run a simple test to see if it works

env = GridWorldStateWrapper(env)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = GridWorldAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)        

from tqdm import tqdm  # Progress bar

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(tuple(obs))

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update(tuple(obs), action, reward, terminated, tuple(next_obs))

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def plot_results(env, agent):
    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # action = agent.get_action(obs)
            action = agent.get_action(tuple(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

# Test your agent
plot_results(env, agent)
test_agent(agent, env)