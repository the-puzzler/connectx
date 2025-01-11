import numpy as np
from numba import jit
import random
import time

@jit(nopython=True)
def check_winner1(board, x, y, player, rows, columns, inarow):
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
def find_winning_move1(board, player, rows, columns, inarow):
    # Check each possible move for a win
    for move in range(columns):
        rows_available = np.where(board[:, move] == 0)[0]
        if rows_available.size == 0:
            continue
        row = rows_available[-1]

        # Try the move
        board[row, move] = player
        if check_winner1(board, row, move, player, rows, columns, inarow):
            board[row, move] = 0  # Undo the move
            return move
        board[row, move] = 0  # Undo the move
    return -1

@jit(nopython=True)
def play_game1(board, move, player, rows, columns, inarow):
    rows_available = np.where(board[:, move] == 0)[0]
    if rows_available.size == 0:
        return None  # Column full
    row = rows_available[-1]
    board[row, move] = player

    if check_winner1(board, row, move, player, rows, columns, inarow):
        return player  # This move led directly to a win

    current_player = 3 - player  # Switch player
    while True:
        # First check for winning move
        winning_move = find_winning_move1(board, current_player, rows, columns, inarow)
        if winning_move != -1:
            rows_available = np.where(board[:, winning_move] == 0)[0]
            row = rows_available[-1]
            board[row, winning_move] = current_player
            return current_player

        # Then check for blocking move
        blocking_move = find_winning_move1(board, 3 - current_player, rows, columns, inarow)
        if blocking_move != -1:
            rows_available = np.where(board[:, blocking_move] == 0)[0]
            row = rows_available[-1]
            board[row, blocking_move] = current_player
        else:
            # If no winning or blocking move, make random move
            possible_moves = np.where(board[0, :] == 0)[0]
            if possible_moves.size == 0:
                return 0  # Draw game
            random_move = np.random.choice(possible_moves)
            rows_available = np.where(board[:, random_move] == 0)[0]
            row = rows_available[-1]
            board[row, random_move] = current_player

        if check_winner1(board, row, blocking_move if blocking_move != -1 else random_move, 
                        current_player, rows, columns, inarow):
            return current_player

        current_player = 3 - current_player  # Switch player

def sim_games1(board, player, rows, columns, inarow):
    valid_moves = [m for m in range(columns) if board[0, m] == 0]
    results = np.zeros(len(valid_moves), dtype=int)
    num_sims = 500

    # First check for immediate winning move
    winning_move = find_winning_move1(board, player, rows, columns, inarow)
    if winning_move != -1:
        return winning_move

    # Then check for immediate blocking move
    blocking_move = find_winning_move1(board, 3 - player, rows, columns, inarow)
    if blocking_move != -1:
        return blocking_move

    # If no immediate winning or blocking move, run simulations
    for idx, move in enumerate(valid_moves):
        sim_score = 0
        for _ in range(num_sims):
            board_copy = board.copy()
            result = play_game1(board_copy, move, player, rows, columns, inarow)
            if result == player:
                sim_score += 1
            elif result == 0:
                continue
            elif result is not None:
                sim_score -= 1
        results[idx] = sim_score

    if results.size > 0:
        best_idx = np.argmax(results)
        best_move = valid_moves[best_idx]
    else:
        best_move = np.random.choice(valid_moves)
    return best_move
#######tet code below







def agent(observation, configuration):
    board = np.array(observation['board']).reshape(configuration['rows'], configuration['columns'])
    player = observation['mark']
    best_move = sim_games1(board, player, configuration['rows'], configuration['columns'], configuration['inarow'])
    return int(best_move)
