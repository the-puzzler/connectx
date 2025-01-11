import numpy as np
from numba import njit, jit, uint8, int32
import random



BUDGET = 50_000_000
N_SIM = 500  # Number of simulations per leaf node
MOVES_PER_GAME = 42

# Formula to calculate the total number of moves for exhaustive + Monte Carlo on leaf nodes
def calculate_total_moves_combined(b, d, n_sim=N_SIM, moves_per_game=MOVES_PER_GAME):
    # Handle edge cases
    if b == 0:  # No valid moves
        return 0
    elif b == 1:  # Only one valid move
        return d + 1 + n_sim * moves_per_game  # Linear path + Monte Carlo on leaf
    else:
        # Normal case: sum of geometric series for exhaustive moves + Monte Carlo simulations
        exhaustive_moves = (b ** (d + 1) - 1) // (b - 1)
        monte_carlo_moves = b ** d * n_sim * moves_per_game
        return exhaustive_moves + monte_carlo_moves

# Function to dynamically calculate maximum depth within budget
def calculate_max_depth(b, budget=BUDGET):
    max_depth = 0
    while calculate_total_moves_combined(b, max_depth) <= budget:
        max_depth += 1
    return max_depth - 1  # Subtract 1 because we go one step too far in the loop

def evaluate_tree(node, rows, columns, inarow, num_sims):
    """
    Recursively evaluates the game tree using minimax principles.
    Performs Monte Carlo simulation on leaf nodes and propagates values upward.
    """
    # Base case: leaf node
    if not node.children:
        # If value is already set (winning position), return it
        if abs(node.value) == float('inf'):
            return node.value
        # Otherwise, evaluate with Monte Carlo
        node.value = monte_carlo_leaf_evaluation(node.board, node.player, rows, columns, inarow, num_sims)
        return node.value

    # Recursive case: evaluate children and propagate values
    child_values = []
    for child in node.children:
        child_value = evaluate_tree(child, rows, columns, inarow, num_sims)
        child_values.append(child_value)

    if node.player == 1:  # Maximizing player
        node.value = max(child_values)
    else:  # Minimizing player
        node.value = min(child_values)
    return node.value

@jit(nopython=True)
def check_winner(board, x, y, player, rows, columns, inarow):
    # Directions: vertical, horizontal, diagonal /, diagonal \
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        for d in [1, -1]:
            step = 1
            while True:
                nx, ny = x + step * dx * d, y + step * dy * d
                if 0 <= nx < rows and 0 <= ny < columns and board[nx, ny] == player:
                    count += 1
                    step += 1
                    if count >= inarow:
                        return True
                else:
                    break
        # Reset count for next direction
    return False

@njit()
def find_winning_move(board, player, rows, columns, inarow):
    """Optimized version of find_winning_move that maintains exact functionality"""
    # Pre-allocate arrays for direction checking
    dx = np.array([1, 0, 1, 1], dtype=int32)
    dy = np.array([0, 1, 1, -1], dtype=int32)
    
    # Check each column for a winning move
    for move in range(columns):
        # Find the first empty row in this column
        for r in range(rows - 1, -1, -1):
            if board[r, move] == 0:
                # Try the move
                board[r, move] = player
                
                # Check for win in all directions
                won = False
                for d in range(4):
                    count = 1
                    # Check both directions
                    for sign in (-1, 1):
                        step = 1
                        while True:
                            nr = r + step * dx[d] * sign
                            nc = move + step * dy[d] * sign
                            if (0 <= nr < rows and 
                                0 <= nc < columns and 
                                board[nr, nc] == player):
                                count += 1
                                if count >= inarow:
                                    won = True
                                    break
                                step += 1
                            else:
                                break
                    if won:
                        break
                
                board[r, move] = 0  # Undo the move
                if won:
                    return move
                break
    return -1

@njit()
def play_game(board, player, rows, columns, inarow):
    """Optimized version of play_game that maintains exact functionality"""
    # Pre-allocate arrays for valid moves to avoid repeated allocations
    valid_moves_array = np.empty(columns, dtype=int32)
    current_player = player
    
    while True:
        # First check for winning move
        winning_move = find_winning_move(board, current_player, rows, columns, inarow)
        
        if winning_move != -1:
            # If there's a winning move, take it
            for r in range(rows - 1, -1, -1):
                if board[r, winning_move] == 0:
                    board[r, winning_move] = current_player
                    return current_player
        
        # Then check for blocking move
        blocking_move = find_winning_move(board, 3 - current_player, rows, columns, inarow)
        
        if blocking_move != -1:
            # If there's a blocking move, take it
            for r in range(rows - 1, -1, -1):
                if board[r, blocking_move] == 0:
                    board[r, blocking_move] = current_player
                    break
        else:
            # Get list of valid moves for random selection
            n_valid = 0
            for m in range(columns):
                if board[0, m] == 0:
                    valid_moves_array[n_valid] = m
                    n_valid += 1
            
            if n_valid == 0:
                return 0  # Draw
                
            # Make random move if no winning or blocking move exists
            move = valid_moves_array[np.random.randint(0, n_valid)]
            for r in range(rows - 1, -1, -1):
                if board[r, move] == 0:
                    board[r, move] = current_player
                    x, y = r, move
                    break

        # Check for a win
        last_move = blocking_move if blocking_move != -1 else move
        for r in range(rows - 1, -1, -1):
            if board[r, last_move] == current_player:
                # Pre-allocate direction arrays for win checking
                dx = np.array([1, 0, 1, 1], dtype=int32)
                dy = np.array([0, 1, 1, -1], dtype=int32)
                
                # Check all directions
                won = False
                for d in range(4):
                    count = 1
                    for sign in (-1, 1):
                        step = 1
                        while True:
                            nr = r + step * dx[d] * sign
                            nc = last_move + step * dy[d] * sign
                            if (0 <= nr < rows and 
                                0 <= nc < columns and 
                                board[nr, nc] == current_player):
                                count += 1
                                if count >= inarow:
                                    won = True
                                    break
                                step += 1
                            else:
                                break
                    if won:
                        return current_player
                break

        current_player = 3 - current_player  # Switch player


@jit(nopython=True)
def monte_carlo_leaf_evaluation(board, player, rows, columns, inarow, num_sims):
    wins = 0
    for _ in range(num_sims):
        board_copy = board.copy()
        result = play_game(board_copy, player, rows, columns, inarow)
        if result == 1:
            wins += 1
        elif result == 2:
            wins -= 1
    return wins / num_sims

class GameNode:
    def __init__(self, board, move, player):
        self.board = board
        self.move = move
        self.player = player
        self.value = 0.0
        self.children = []


