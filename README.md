# ConnectX Agent - Minimax with Monte Carlo Leaf Evaluation

## Overview
This repository contains an implementation of a game-playing agent for the ConnectX competition on Kaggle. The agent uses a hybrid approach combining minimax tree search with Monte Carlo simulations at leaf nodes, optimized using Numba for high performance. This implementation achieved a top 10 ranking on the Kaggle ConnectX competition leaderboard.

## Key Features
- Hybrid search strategy combining deterministic and probabilistic approaches
- Dynamic depth calculation based on branching factor and computational budget
- Numba-optimized game simulation and evaluation
- Board size agnostic implementation
- Efficient winning move detection
- Smart move prioritization (winning moves > blocking moves > random moves)

## Technical Implementation
The agent employs several key components:

1. **Tree Search**
  - Minimax search with dynamic depth calculation
  - Efficient pruning of game tree based on computational budget
  - Monte Carlo simulation at leaf nodes

2. **Performance Optimization**
  - Numba JIT compilation for critical functions
  - Pre-allocated arrays for move validation
  - Optimized win-checking algorithms

3. **Game Logic**
  - Board state management
  - Move validation
  - Win detection in all directions (horizontal, vertical, diagonal)

## Key Parameters
- `BUDGET`: 50,000,000 (computation budget)
- `N_SIM`: 500 (number of Monte Carlo simulations per leaf node)
- `MOVES_PER_GAME`: 42 (maximum moves in a standard game)

## Usage
The agent can be used with any board size configuration of ConnectX, though it's optimized for the standard 6x7 board with 4-in-a-row winning condition.

## Performance
The implementation achieves strong performance through:
- Fast move generation and evaluation
- Efficient memory usage
- Balanced exploration of the game tree
- Smart resource allocation between tree search and Monte Carlo simulation
- Top 10 ranking on Kaggle ConnectX competition leaderboard

## Dependencies
- NumPy
- Numba
