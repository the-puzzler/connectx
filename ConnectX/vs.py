import numpy as np
import random
from tqdm import tqdm
from numba import jit
import time
import multiprocessing as mp
from functools import partial
from collections import defaultdict

@jit(nopython=True)
def check_winner(board, x, y, player, rows, columns, inarow):
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
    return False

def play_single_game(args):
    """
    Simulates a single game with the given parameters.
    Returns additional statistics about the game.
    """
    rows, columns, inarow, name1, name2, game_index = args
    
    agent_module1 = __import__(name1)
    agent_module2 = __import__(name2)
    
    bot1_agent = agent_module1.agent
    bot2_agent = agent_module2.agent
    
    board = np.zeros((rows, columns), dtype=int)
    # Alternate starting player based on game index
    starting_player = 1 if game_index % 2 == 0 else 2
    current_player = starting_player
    bot1_move_times = []
    bot2_move_times = []
    move_sequence = []
    num_moves = 0

    while True:
        possible_moves = np.where(board[0, :] == 0)[0]
        if possible_moves.size == 0:
            return {
                'winner': 0,
                'starting_player': starting_player,
                'num_moves': num_moves,
                'bot1_move_times': bot1_move_times,
                'bot2_move_times': bot2_move_times,
                'move_sequence': move_sequence
            }

        observation = {'board': board.flatten(), 'mark': current_player}
        configuration = {'rows': rows, 'columns': columns, 'inarow': inarow}

        if current_player == 1:
            start_time = time.time()
            move = bot1_agent(observation, configuration)
            end_time = time.time()
            bot1_move_times.append(end_time - start_time)
        else:
            start_time = time.time()
            move = bot2_agent(observation, configuration)
            end_time = time.time()
            bot2_move_times.append(end_time - start_time)

        rows_available = np.where(board[:, move] == 0)[0]
        if rows_available.size == 0:
            continue
            
        row = rows_available[-1]
        board[row, move] = current_player
        move_sequence.append(move)
        num_moves += 1

        if check_winner(board, row, move, current_player, rows, columns, inarow):
            return {
                'winner': current_player,
                'starting_player': starting_player,
                'num_moves': num_moves,
                'bot1_move_times': bot1_move_times,
                'bot2_move_times': bot2_move_times,
                'move_sequence': move_sequence
            }

        current_player = 3 - current_player

def simulate_multiple_games_parallel(num_games, rows=6, columns=7, inarow=4, name1='agent_exhaust_monty2_ab', name2='agent_monty2'):
    """
    Simulate multiple games in parallel with enhanced statistics reporting.
    Ensures balanced first-mover opportunities.
    """
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)
    
    # Make sure num_games is even to ensure balanced opportunities
    if num_games % 2 != 0:
        num_games += 1
        print(f"Adjusted number of games to {num_games} to ensure balanced opportunities")
    
    # Create game parameters with game index for alternating first player
    game_params = [(rows, columns, inarow, name1, name2, i) for i in range(num_games)]

    results = list(tqdm(
        pool.imap(play_single_game, game_params),
        total=num_games,
        desc=f"Simulating {num_games} games using {num_cores} cores"
    ))

    pool.close()
    pool.join()

    # Initialize statistics
    stats = {
        'total_games': num_games,
        'bot1': {
            'total_wins': 0,
            'wins_as_first': 0,
            'wins_as_second': 0,
            'times_as_first': 0,
            'times_as_second': 0,
            'avg_moves_to_win': [],
            'move_times': defaultdict(list)
        },
        'bot2': {
            'total_wins': 0,
            'wins_as_first': 0,
            'wins_as_second': 0,
            'times_as_first': 0,
            'times_as_second': 0,
            'avg_moves_to_win': [],
            'move_times': defaultdict(list)
        },
        'draws': 0,
        'avg_game_length': [],
        'common_opening_moves': defaultdict(int)
    }

    # Process results
    for game in results:
        winner = game['winner']
        starting_player = game['starting_player']
        num_moves = game['num_moves']
        stats['avg_game_length'].append(num_moves)

        # Track number of times each bot goes first/second
        if starting_player == 1:
            stats['bot1']['times_as_first'] += 1
            stats['bot2']['times_as_second'] += 1
        else:
            stats['bot1']['times_as_second'] += 1
            stats['bot2']['times_as_first'] += 1

        # Record opening moves
        if len(game['move_sequence']) > 0:
            stats['common_opening_moves'][game['move_sequence'][0]] += 1

        # Process move times
        for move_num, time_taken in enumerate(game['bot1_move_times']):
            stats['bot1']['move_times'][move_num].append(time_taken)
        for move_num, time_taken in enumerate(game['bot2_move_times']):
            stats['bot2']['move_times'][move_num].append(time_taken)

        if winner == 0:
            stats['draws'] += 1
        elif winner == 1:
            stats['bot1']['total_wins'] += 1
            stats['bot1']['avg_moves_to_win'].append(num_moves)
            if starting_player == 1:
                stats['bot1']['wins_as_first'] += 1
            else:
                stats['bot1']['wins_as_second'] += 1
        else:  # winner == 2
            stats['bot2']['total_wins'] += 1
            stats['bot2']['avg_moves_to_win'].append(num_moves)
            if starting_player == 2:
                stats['bot2']['wins_as_first'] += 1
            else:
                stats['bot2']['wins_as_second'] += 1

    # Print comprehensive statistics
    print("\n=== Connect Four Tournament Results ===")
    print(f"\nTotal Games Played: {num_games} using {num_cores} CPU cores")
    
    print("\n=== Overall Results ===")
    print(f"{name1} wins: {stats['bot1']['total_wins']} ({(stats['bot1']['total_wins'] / num_games) * 100:.2f}%)")
    print(f"{name2} wins: {stats['bot2']['total_wins']} ({(stats['bot2']['total_wins'] / num_games) * 100:.2f}%)")
    print(f"Draws: {stats['draws']} ({(stats['draws'] / num_games) * 100:.2f}%)")

    print("\n=== First/Second Player Analysis ===")
    print(f"{name1}:")
    print(f"  Times as first player: {stats['bot1']['times_as_first']}")
    print(f"  Times as second player: {stats['bot1']['times_as_second']}")
    print(f"  Wins as first player: {stats['bot1']['wins_as_first']} ({(stats['bot1']['wins_as_first'] / stats['bot1']['times_as_first'] * 100) if stats['bot1']['times_as_first'] > 0 else 0:.2f}% win rate as first)")
    print(f"  Wins as second player: {stats['bot1']['wins_as_second']} ({(stats['bot1']['wins_as_second'] / stats['bot1']['times_as_second'] * 100) if stats['bot1']['times_as_second'] > 0 else 0:.2f}% win rate as second)")
    
    print(f"{name2}:")
    print(f"  Times as first player: {stats['bot2']['times_as_first']}")
    print(f"  Times as second player: {stats['bot2']['times_as_second']}")
    print(f"  Wins as first player: {stats['bot2']['wins_as_first']} ({(stats['bot2']['wins_as_first'] / stats['bot2']['times_as_first'] * 100) if stats['bot2']['times_as_first'] > 0 else 0:.2f}% win rate as first)")
    print(f"  Wins as second player: {stats['bot2']['wins_as_second']} ({(stats['bot2']['wins_as_second'] / stats['bot2']['times_as_second'] * 100) if stats['bot2']['times_as_second'] > 0 else 0:.2f}% win rate as second)")

    print("\n=== Game Length Analysis ===")
    print(f"Average game length: {sum(stats['avg_game_length']) / len(stats['avg_game_length']):.2f} moves")
    if stats['bot1']['avg_moves_to_win']:
        print(f"Average moves for {name1} wins: {sum(stats['bot1']['avg_moves_to_win']) / len(stats['bot1']['avg_moves_to_win']):.2f}")
    if stats['bot2']['avg_moves_to_win']:
        print(f"Average moves for {name2} wins: {sum(stats['bot2']['avg_moves_to_win']) / len(stats['bot2']['avg_moves_to_win']):.2f}")

    print("\n=== Performance Analysis ===")
    # Calculate average move times for first 5 moves
    for bot_num, bot_name in [(1, name1), (2, name2)]:
        print(f"\n{bot_name} move timing (first 5 moves):")
        bot_key = f'bot{bot_num}'
        for move_num in range(5):
            if move_num in stats[bot_key]['move_times']:
                avg_time = sum(stats[bot_key]['move_times'][move_num]) / len(stats[bot_key]['move_times'][move_num])
                print(f"  Move {move_num + 1}: {avg_time:.5f} seconds")

    print("\n=== Opening Move Analysis ===")
    total_opening_moves = sum(stats['common_opening_moves'].values())
    print("Most common opening moves (column numbers):")
    sorted_openings = sorted(stats['common_opening_moves'].items(), key=lambda x: x[1], reverse=True)
    for column, count in sorted_openings[:3]:
        print(f"  Column {column}: {count} times ({(count/total_opening_moves)*100:.2f}%)")

if __name__ == '__main__':
    simulate_multiple_games_parallel(num_games=100, name1='agent_monty2wip', name2='agent_monty2')