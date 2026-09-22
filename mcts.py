import numpy as np
import random
from tic_tac_toe import terminal_check, legal_actions, next_state, rollout


# The class that we will use to store the tree
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.visits = 0
        self.value_sum = 0.0

    # Dynamically calculate the exploitation (sometimes referred to as Q(node))
    @property
    def exploitation(self):
        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits

    # The UCT formula (with its parameter c)
    def UCT(self, c):
        if self.visits == 0:
            return np.inf

        Q = self.exploitation
        E = c * np.sqrt(np.log(self.parent.visits) / self.visits)
        return Q + E

    # Expand the leaf node
    def expand(self, action, new_state):
        child = Node(
            state=new_state,
            parent=self,
            action=action,
        )

        self.children.append(child)

        return child

    # Select the child with the highest UCT value
    def select_child(self, c):
        best_child = None
        best_UCT = -np.inf

        for child in self.children:
            child_UCT = child.UCT(c)

            if child_UCT > best_UCT:
                best_child = child
                best_UCT = child_UCT

        return best_child


# Propagate the reward back from the node (with alternating sign for the two player tic-tac-toe game)
def backpropagate(node, reward):
    while node is not None:
        # Increase the visit count
        node.visits += 1

        # Update the value sum
        node.value_sum += reward

        # Go back up to the parent node
        node = node.parent

        # Alternate the sign of the reward as the players switch
        reward = -reward


# The final MCTS function that will determine the player action at a given board state
def mcts(board_state, player, num_simulations, c):
    root = Node(board_state)

    for _ in range(num_simulations):
        # Start at the root for each simulation
        node = root

        # Start at the current player for each simulation
        current_player = player

        # Traverse the tree until you find a node to expand
        while terminal_check(node.state) is None and len(node.children) == len(
            legal_actions(node.state)
        ):
            # Select the child with the highest UCT
            node = node.select_child(c)

            # Switch player
            current_player *= -1

        # Expand the node if it is not terminal
        if terminal_check(node.state) is None:
            # The legal moves
            legal = legal_actions(node.state)

            # The tried moves
            tried = [child.action for child in node.children]

            # The legal moves that haven't been tried yet
            untried = list(set(legal) - set(tried))

            # Randomly select one of these
            action = random.choice(untried)

            new_state = next_state(node.state, action, current_player)

            node = node.expand(action, new_state)

            # Switch player
            current_player *= -1

        # Check the rollout to see what reward we should apply (to update the values)
        result = rollout(node.state, current_player)

        # We use this for the reward
        reward = result * (-current_player)

        backpropagate(node, reward)

    # Select the action at the root with the largest visit count
    best_action = None
    most_visits = -1

    for child in root.children:
        if child.visits > most_visits:
            best_action = child.action
            most_visits = child.visits

    return best_action
