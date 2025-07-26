
import gymnasium as gym
import numpy as np

class GridWorldStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Access the original unwrapped environment
        size = env.unwrapped.size
        self.observation_space = gym.spaces.Box(
            low=0, high=size-1, shape=(4,), dtype=int
        )

    def observation(self, observation):
        # Convert {"agent": [x,y], "target": [x,y]} -> (ax, ay, tx, ty)
        return np.array([
            observation["agent"][0],
            observation["agent"][1],
            observation["target"][0],
            observation["target"][1]
        ], dtype=int)