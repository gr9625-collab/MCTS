import numpy as np

from mcts import mcts
from tic_tac_toe import terminal_check, minimax_action

# ==================================================
# PARAMETERS
# ==================================================

num_games = 100
num_simulations = 1000
c = 1.5


# ==================================================
# BASIC TEST STATES
# ==================================================

print("=" * 50)
print("BASIC TEST STATES")
print("=" * 50)

# Three test states to see what move MCTS makes
test_states = {
    "immediate_win": np.array(
        [
            [1, 1, 0],
            [-1, -1, 0],
            [0, 0, 0],
        ]
    ),
    "must_block": np.array(
        [
            [-1, -1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]
    ),
    "take_center": np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
        ]
    ),
}

# Check what action MCTS makes (here it is player 1's turn)
for name, state in test_states.items():
    print(name)
    print(state)

    action = mcts(
        state,
        player=1,
        num_simulations=num_simulations,
        c=c,
    )

    print("MCTS chose:", action)
    print("=" * 50)


# ==================================================
# MCTS VS MINIMAX
# ==================================================

print("=" * 50)
print("MCTS VS MINIMAX")
print("=" * 50)

# Get MCTS and Minimax to play games against each other and check the results
mcts_wins = 0
minimax_wins = 0
draws = 0

for i in range(num_games):
    current_board = np.zeros((3, 3))

    # The player with value 1 always goes first
    # This alternates between MCTS and Minimax for even and odd games
    player = 1

    # Play the game until it finishes
    while terminal_check(current_board) is None:

        if i % 2 == 0:
            # Player 1 is MCTS and player -1 is Minimax
            if player == 1:
                action = mcts(
                    current_board,
                    player,
                    num_simulations=num_simulations,
                    c=c,
                )
            else:
                action = minimax_action(current_board, player)

        else:
            # Player 1 is Minimax and player -1 is MCTS
            if player == -1:
                action = mcts(
                    current_board,
                    player,
                    num_simulations=num_simulations,
                    c=c,
                )
            else:
                action = minimax_action(current_board, player)

        # The current player makes their move
        current_board[action] = player

        # Switch players
        player *= -1

    result = terminal_check(current_board)

    # Increase the win counts if a win occurred
    if i % 2 == 0:
        if result == 1:
            mcts_wins += 1
        elif result == -1:
            minimax_wins += 1
    else:
        if result == 1:
            minimax_wins += 1
        elif result == -1:
            mcts_wins += 1

    # Otherwise increase the draw count
    if result == 0:
        draws += 1

    print(f"Games completed: {i + 1}")


# ==================================================
# RESULTS
# ==================================================

print("=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Games:        {num_games}")
print(f"MCTS wins:    {mcts_wins}")
print(f"Minimax wins: {minimax_wins}")
print(f"Draws:        {draws}")
