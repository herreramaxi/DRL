# import gymnasium as gym



# print([env.id for env in gym.envs.registry])

# # Register the environment
# register_chess_env()

# # Create environment
# env = gym.make("gymnasium_env/ChessGame-v0")

# print("Action space:", env.action_space)
# print("Observation space:", env.observation_space)

# # Reset environment
# obs = env.reset()
# print("Initial observation:", obs)

# done = False
# steps = 0

# while not done and steps < 10:  # Take 10 steps max for testing
#     action = env.action_space.sample()  # Random action
#     obs, reward, done, info = env.step(action)
#     print(f"Step {steps + 1}: Action={action}, Reward={reward}, Done={done}")
#     print("Observation:", obs)
#     steps += 1

# env.close()

import random
from ChessGame.ChessEnv import register_chess_env
import gymnasium as gym


register_chess_env()
# env = gym.make("gymnasium_env/ChessGame-v0")

# obs, info = env.reset()
# unwrapped = env.unwrapped

# env.render()  # <-- Show initial board

# for _ in range(5):
#     legal_moves = unwrapped.get_legal_moves()
#     action = random.choice(legal_moves)  # pick only valid moves
#     obs, reward, terminated, truncated, info = env.step(action)
#     unwrapped.render()  # <-- Show board after each move
#     print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
#     if terminated or truncated:
#         break

# env.close()



env = gym.make("gymnasium_env/ChessGame-v0")

obs, info = env.reset()
unwrapped = env.unwrapped


env.render()  # <-- Show initial board
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for _ in range(2):
    legal_moves = unwrapped.get_legal_moves()
    legal_moves = list(legal_moves)  # Convert set to list
    if not legal_moves:
        print("No legal moves available.")
        break
    action = random.choice(legal_moves)  # Pick a valid move
    obs, reward, terminated, truncated, info = env.step(action)
    unwrapped.render()  # <-- Show board after each move
    print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    if terminated or truncated:
        break

env.close()