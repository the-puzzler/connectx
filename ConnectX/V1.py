import numpy as np
from numba import jit
import random
import time

@jit(nopython=True)
def check_winner(board, x, y, player, rows, columns, inarow):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        # Check forward
        count = 1
        for step in range(1, inarow):
            nx, ny = x + step * dx, y + step * dy
            if 0 <= nx < rows and 0 <= ny < columns and board[nx, ny] == player:
                count += 1
            else:
                break

        if count >= inarow:
                return True

        # Check backward
        for step in range(1, inarow):
            nx, ny = x - step * dx, y - step * dy
            if 0 <= nx < rows and 0 <= ny < columns and board[nx, ny] == player:
                count += 1
            else:
                break

        # Only need to check if count is at least `inarow`
        if count >= inarow:
            return True

    return False

@jit(nopython=True)
def play_game(board, move, player, rows, columns, inarow):
    rows_available = np.where(board[:, move] == 0)[0]
    if rows_available.size == 0:
        return None  # Column full
    row = rows_available[-1]
    board[row, move] = player

    if check_winner(board, row, move, player, rows, columns, inarow):
        return player  # This move led directly to a win

    current_player = 3 - player  # Switch player
    while True:
        possible_moves = np.where(board[0, :] == 0)[0]
        if possible_moves.size == 0:
            return 0  # Draw game

        random_move = np.random.choice(possible_moves)  # Use np.random.choice
        rows_available = np.where(board[:, random_move] == 0)[0]
        random_row = rows_available[-1]
        board[random_row, random_move] = current_player

        if check_winner(board, random_row, random_move, current_player, rows, columns, inarow):
            return current_player  # This player wins

        current_player = 3 - current_player  # Switch player



def sim_games(board, player, rows, columns, inarow):
    valid_moves = [m for m in range(columns) if board[0, m] == 0]
    results = np.zeros(len(valid_moves), dtype=int)
    print(results)
    num_sims = 20_000
    for idx, move in enumerate(valid_moves):
        sim_score = 0
        for _ in range(num_sims):
            board_copy = board.copy()
            result = play_game(board_copy, move, player, rows, columns, inarow)
            if result == player:
                sim_score += 1
            elif result == 0:
                continue
            elif result is not None:
                sim_score -= 1
        results[idx] = sim_score
    print(results)
    if results.size > 0:
        best_idx = np.argmax(results)
        best_move = valid_moves[best_idx]
    else:
        print('used random')
        best_move = np.random.choice(valid_moves)  # Use np.random.choice
    return best_move

def agent(observation, configuration):
    
    board = np.array(observation['board']).reshape(configuration['rows'], configuration['columns'])
    player = observation['mark']
    import time
    t1 = time.time()
    best_move = sim_games(board, player, configuration['rows'], configuration['columns'], configuration['inarow'])
    
    print(time.time() - t1)
    print(best_move)
    print('-'*26)
    return int(best_move)
