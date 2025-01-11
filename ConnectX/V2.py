import numpy as np
from numba import njit
import random

BUDGET = 50_000_000
N_SIM = 700  # Number of simulations per leaf node
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

@njit
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

@njit
def play_game(board, player, rows, columns, inarow):
    current_player = player
    while True:
        # Get list of valid moves
        valid_moves = []
        for m in range(columns):
            if board[0, m] == 0:
                valid_moves.append(m)
        if len(valid_moves) == 0:
            return 0  # Draw
        move = valid_moves[np.random.randint(0, len(valid_moves))]

        # Place the piece
        for r in range(rows - 1, -1, -1):
            if board[r, move] == 0:
                board[r, move] = current_player
                x, y = r, move
                break

        # Check for a win
        if check_winner(board, x, y, current_player, rows, columns, inarow):
            return current_player

        current_player = 3 - current_player  # Switch player

@njit
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

def build_game_tree(board, depth, player, rows, columns, inarow, move=None):
    node = GameNode(board.copy(), move, player)

    # Base case: reached depth limit or game over
    if depth == 0:
        return node

    valid_moves = [m for m in range(columns) if board[0, m] == 0]

    # Game over check
    if not valid_moves:
        return node

    # Recursively build children
    for move in valid_moves:
        board_copy = board.copy()
        for r in range(rows - 1, -1, -1):
            if board_copy[r, move] == 0:
                board_copy[r, move] = player
                x, y = r, move
                break

        # Check if this move results in a win
        if check_winner(board_copy, x, y, player, rows, columns, inarow):
            child = GameNode(board_copy, move, 3 - player)
            child.value = float('inf') if player == 1 else float('-inf')
            node.children.append(child)
            continue

        # Recurse
        child = build_game_tree(board_copy, depth - 1, 3 - player, rows, columns, inarow, move)
        node.children.append(child)

    return node

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

def agent(observation, configuration):
    board = np.array(observation['board']).reshape(configuration['rows'], configuration['columns'])
    player = observation['mark']
    rows = configuration['rows']
    columns = configuration['columns']
    inarow = configuration['inarow']

    # Calculate branching factor and maximum depth
    b = np.count_nonzero(board[0, :] == 0)
    max_depth = calculate_max_depth(b)
    # print(max_depth)
    # Build the game tree
    root = build_game_tree(board, max_depth, player, rows, columns, inarow)

    # Evaluate the tree
    num_sims = N_SIM  # You can adjust this based on your needs
    evaluate_tree(root, rows, columns, inarow, num_sims)

    # Choose the best move based on children's values
    best_value = float('-inf') if player == 1 else float('inf')
    best_move = None
    for child in root.children:
        if (player == 1 and child.value > best_value) or (player == 2 and child.value < best_value):
            best_value = child.value
            best_move = child.move
    # print('Player: ', player)
    # print(len(root.children))
    # print(best_move)
    # print(best_value)
    # for child in root.children:
    #   print(child.value)
    # print('-'*15)
    # If best_move is still None, select the first available move
    if best_move is None:
        valid_moves = [c for c in range(columns) if board[0, c] == 0]
        best_move = valid_moves[0] if valid_moves else 0  # Default to column 0 if no valid moves


    return int(best_move)
