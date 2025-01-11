import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from agent_exhaust_monty2 import build_game_tree, evaluate_tree

# Define constants
DEPTH_INITIAL_MOVES = 5  # Depth for initial moves (up to 5 tokens on board)
DEPTH_ADDITIONAL_MOVES = 6  # Additional depth for exhaustive search
N_SIM = 100  # Number of Monte Carlo simulations per leaf node
OPENING_BOOK_PATH = "opening_book.pkl"  # Path to store the opening book

# Function to hash the board state (as a tuple for immutability)
def hash_board(board):
    return tuple(board.flatten())

# Function to generate all game states for a specific move count
def generate_initial_states(board, depth, player, rows, columns, inarow, states):
    if depth == 0:
        states.append((board.copy(), player))  # Append the state and player for evaluation
        return

    # Generate child nodes by simulating each valid move
    valid_moves = [c for c in range(columns) if board[0, c] == 0]
    for move in valid_moves:
        board_copy = board.copy()
        for r in range(rows - 1, -1, -1):
            if board_copy[r, move] == 0:
                board_copy[r, move] = player
                break
        generate_initial_states(board_copy, depth - 1, 3 - player, rows, columns, inarow, states)

# Function to evaluate a single game state by exploring deeper moves
def evaluate_state(task):
    board, player, rows, columns, inarow = task
    max_depth = DEPTH_ADDITIONAL_MOVES
    root = build_game_tree(board, max_depth, player, rows, columns, inarow)
    evaluate_tree(root, rows, columns, inarow, N_SIM)

    # Choose the best move based on evaluation
    best_value = float('-inf') if player == 1 else float('inf')
    best_move = None
    for child in root.children:
        if (player == 1 and child.value > best_value) or (player == 2 and child.value < best_value):
            best_value = child.value
            best_move = child.move

    board_key = hash_board(board)
    return board_key, best_move

# Core function to build and save the opening book in parallel
def build_and_save_opening_book_parallel(rows, columns, inarow):
    # Generate initial board states up to DEPTH_INITIAL_MOVES
    initial_board = np.zeros((rows, columns), dtype=int)
    initial_states = []
    generate_initial_states(initial_board, DEPTH_INITIAL_MOVES, 1, rows, columns, inarow, initial_states)
    print(f"Generated {len(initial_states)} initial states for parallel evaluation.")

    # Prepare tasks for parallel processing (each task is a board state and metadata)
    tasks = [(board, player, rows, columns, inarow) for board, player in initial_states]
    
    # Use multiprocessing to evaluate states in parallel
    opening_book = {}
    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=len(tasks), desc="Evaluating States") as progress_bar:
            for board_key, best_move in pool.imap_unordered(evaluate_state, tasks):
                opening_book[board_key] = best_move
                progress_bar.update(1)

    # Save the opening book to a file
    with open(OPENING_BOOK_PATH, 'wb') as f:
        pickle.dump(opening_book, f)
    print("Opening book created and saved.")

# Ensure the script only runs in main context
if __name__ == "__main__":
    build_and_save_opening_book_parallel(6, 7, 4)  # Build the opening book for Connect Four
