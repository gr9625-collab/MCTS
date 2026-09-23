import numpy as np
import random


# Get the legal actions
def legal_actions(board_state):
    actions = []
    for i in range(3):
        for j in range(3):
            if board_state[i, j] == 0:
                actions.append((i, j))
    return actions


# Update a board state with a given action (alters the board state)
def make_action(board_state, action, player):
    legal = legal_actions(board_state)
    if action not in legal:
        # We can deal with the exact course of action later, or exact error message
        return None
    # Player 1 is represented by 1 and player 2 by -1
    board_state[action] = player
    return board_state


# Returns the new state after a given action (does NOT update the board state)
def next_state(board_state, action, player):
    new_state = np.copy(board_state)
    new_state = make_action(new_state, action, player)
    return new_state


# Check if the game has ended (i.e. a player has won or the board is full anyway)
# This assumes the board is in a valid state
def terminal_check(board_state):
    # Check for constant rows
    for i in range(3):
        if (
            board_state[i, 0] != 0
            and board_state[i, 0] == board_state[i, 1]
            and board_state[i, 1] == board_state[i, 2]
        ):
            return board_state[i, 0]

    # Check for constant columns
    for i in range(3):
        if (
            board_state[0, i] != 0
            and board_state[0, i] == board_state[1, i]
            and board_state[1, i] == board_state[2, i]
        ):
            return board_state[0, i]

    # Check for constant "up-diagonal"
    if (
        board_state[2, 0] != 0
        and board_state[2, 0] == board_state[1, 1]
        and board_state[1, 1] == board_state[0, 2]
    ):
        return board_state[2, 0]

    # Check for constant "down-diagonal"
    if (
        board_state[0, 0] != 0
        and board_state[0, 0] == board_state[1, 1]
        and board_state[1, 1] == board_state[2, 2]
    ):
        return board_state[0, 0]

    # If there are no winning states and the board is full return a draw
    if legal_actions(board_state) == []:
        return 0

    # Otherwise return that the game is ongoing
    return None


# Randomly play moves and see which player wins from a given state (it does NOT update the board state)
def rollout(board_state, player):
    state = np.copy(board_state)

    while terminal_check(state) is None:
        # Choose a random legal action
        actions = legal_actions(state)
        action = random.choice(actions)

        # Make this action
        state = make_action(state, action, player)

        # Switch player
        player *= -1

    # Return the final game state value
    return terminal_check(state)


# ==================================================
# MINIMAX PLAYER (for testing against MCTS)
# ==================================================


# Computing the minimax value
def minimax(board_state, player):
    if terminal_check(board_state) is not None:
        return terminal_check(board_state)

    values = []

    for action in legal_actions(board_state):
        new_state = next_state(board_state, action, player)

        value = minimax(new_state, -player)
        values.append(value)

    if player == 1:
        return max(values)
    else:
        return min(values)


# Using minimax to decide the next move
def minimax_action(board_state, player):
    actions = legal_actions(board_state)

    best_action = None

    if player == 1:
        best_value = -np.inf

        for action in actions:
            new_state = next_state(board_state, action, player)
            value = minimax(new_state, -player)

            if value > best_value:
                best_value = value
                best_action = action

    else:
        best_value = np.inf

        for action in actions:
            new_state = next_state(board_state, action, player)
            value = minimax(new_state, -player)

            if value < best_value:
                best_value = value
                best_action = action

    return best_action
