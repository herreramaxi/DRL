
from collections import defaultdict
import gymnasium as gym
import numpy as np

class GridWorldAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, state: tuple[int, int, int, int]) -> int:
        """
        Choose an action using an epsilon-greedy strategy.

        Args:
            state: Current state represented as a tuple (agent_x, agent_y, target_x, target_y).

        Returns:
            action: Integer in [0, 3] (0=right, 1=up, 2=left, 3=down)
        """
        # Exploration: choose random action
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Exploitation: choose best known action
        return int(np.argmax(self.q_values[state]))


    def update(
        self,
        state: tuple[int, int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple[int, int, int, int],
    ):
        """
        Update the Q-value for the given state-action pair based on observed reward and next state.

        Args:
            state: Current state as a tuple (agent_x, agent_y, target_x, target_y)
            action: Action taken (0=right, 1=up, 2=left, 3=down)
            reward: Reward received after taking the action
            terminated: Whether the episode ended after this step
            next_state: The next state as a tuple (agent_x, agent_y, target_x, target_y)
        """
        # Best Q-value for next state (0 if terminal)
        future_q_value = 0 if terminated else np.max(self.q_values[next_state])

        # Bellman target
        target = reward + self.discount_factor * future_q_value

        # Temporal difference error
        td_error = target - self.q_values[state][action]

        # Update rule
        self.q_values[state][action] += self.lr * td_error

        # Track learning error
        self.training_error.append(td_error)


    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

