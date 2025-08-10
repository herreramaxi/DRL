import time
import numpy as np
from ChessGame.ChessEnv import MinichessEnv
from ChessGame.games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

def run_random_game(env, max_moves=100):
    # Reset the environment (gymnasium API returns obs, info)
    env = GardnerMiniChessGame()
    board = env.getInitBoard()
    # env.reset()
    moves = 0
    start = time.perf_counter()
    done = False
    truncated = False
    player = 1
    while moves < max_moves and not (done or truncated):
        legal_moves = env.getValidMoves(board, player, return_type="list")    
        move = np.random.choice(legal_moves)                
        board, player = env.getNextState(board, player, move)
        game_result = env.getGameEnded(board, 1, 0.5)
        done = game_result != 0
        moves += 1

    end = time.perf_counter()
    return moves, end - start

def benchmark(num_games=10, max_moves=100):
    env = GardnerMiniChessGame()
    total_moves = 0
    total_time = 0.0

    for i in range(1, num_games + 1):
        moves, duration = run_random_game(env, max_moves)
        total_moves += moves
        total_time += duration
        print(f"Game {i:2d}: {moves:3d} moves in {duration:.4f}s --> {moves/duration:6.0f} moves/sec")

    # env.close()
    print("-" * 50)
    overall_mps = total_moves / total_time if total_time > 0 else 0
    print(f"Overall: {total_moves} moves in {total_time:.4f}s --> {overall_mps:.0f} moves/sec")


def benchmark2(max_moves=100):
    env = MinichessEnv()
    total_moves = 0
    total_time = 0.0
    moves= 0
   

    start = time.perf_counter()
    done = False
    truncated = False
    games = 0
    while moves < max_moves: 
        env.reset()
        games +=1
        while not (done or truncated):
            # Sample a random legal action via action_masks()
            mask = env.action_masks()               # 1-D bool array of legal actions
            candidates = np.nonzero(mask)[0]        # indices of True entries
            action = int(np.random.choice(candidates))
            
            # Step through the env (gymnasium: obs, reward, done, truncated, info)
            obs, reward, done, truncated, info = env.step(action)
            moves += 1

    end = time.perf_counter()
    total_time = end -start
    overall_mps = moves / total_time if total_time > 0 else 0
    print(f"Games: {games}, Overall: {moves} moves in {total_time:.4f}s --> {overall_mps:.0f} moves/sec")

 
if __name__ == "__main__":
    benchmark(num_games=10, max_moves=100)
    # benchmark2(15)
