import chess
import random
import time
import pandas as pd

def run_random_game(max_moves=100):
    board = chess.Board()
    moves = 0
    start = time.perf_counter()
    while moves < max_moves and not board.is_game_over():
        move = random.choice(list(board.legal_moves))
        board.push(move)
        moves += 1
    end = time.perf_counter()
    return moves, end - start

def benchmark(num_games=10, max_moves=100):
    games = []
    for i in range(num_games):
        moves, duration = run_random_game(max_moves)
        games.append({
            "Game": i + 1,
            "Moves": moves,
            "Duration (s)": duration,
            "Moves per Second": moves / duration
        })

    df = pd.DataFrame(games)
    print(df)
    total_moves = df["Moves"].sum()
    total_time = df["Duration (s)"].sum()
    overall_mps = total_moves / total_time
    print(f"Total moves: {total_moves}, Total time: {total_time:.4f}s, Overall moves/sec: {overall_mps:.0f}")


def benchmark2(max_moves=100):
    board = chess.Board()
    total_moves = 0
    total_time = 0.0
    moves= 0
   

    start = time.perf_counter()
    done = False
    truncated = False
    games = 0
    while moves < max_moves: 
        board.reset()
        games +=1
        while not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)
            moves += 1

    end = time.perf_counter()
    total_time = end -start
    overall_mps = moves / total_time if total_time > 0 else 0
    print(f"Games: {games}, Overall: {moves} moves in {total_time:.4f}s --> {overall_mps:.0f} moves/sec")
if __name__ == "__main__":
    benchmark(num_games=10, max_moves=100)
    # benchmark2(1000)