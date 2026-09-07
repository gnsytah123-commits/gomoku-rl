"""Classic UCT search with random rollouts. Used as the first teacher."""

import math
import random
import numpy as np
from gomoku_class import Gomoku


class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0


class MCTS:
    def __init__(self, game, c=0.8, iterations=2000):
        self.root = Node(game.clone())
        self.c = c
        self.root_player = game.player
        self.iterations = iterations

    def UCT(self, node):
        if node.N == 0 or node.parent is None or node.parent.N == 0:
            return float("inf")
        return node.Q + self.c * math.sqrt(math.log(node.parent.N) / node.N)

    def selection(self, node):
        cur = node
        while (
            cur.children
            and len(cur.children) == len(cur.state.legal_moves())
            and cur.state.status == cur.state.ONGOING
        ):
            cur = max(cur.children.values(), key=lambda child: self.UCT(child))
        return cur

    def expand(self, node):
        unexpanded = [m for m in node.state.legal_moves() if m not in node.children]
        if not unexpanded:
            return node

        move = random.choice(unexpanded)
        new_state = node.state.clone()
        new_state.make(move)
        child = Node(new_state, parent=node, move=move)
        node.children[move] = child
        return child

    def simulation(self, node):
        state = node.state.clone()
        while state.status == state.ONGOING:
            state.make(random.choice(state.legal_moves()))

        winner = state.winner()
        if winner is None:
            return 0

        # Q is from the player who moved to this node (the parent).
        if node.parent is None:
            mover = state.other(state.player)
        else:
            mover = node.parent.state.player
        return 1 if winner == mover else -1

    def back_propagate(self, node, reward):
        cur = node
        r = reward
        while cur is not None:
            cur.N += 1
            cur.W += r
            cur.Q = cur.W / cur.N
            r = -r
            cur = cur.parent

    def _run_search(self, game):
        self.root = Node(game.clone())
        self.root_player = game.player

        for _ in range(self.iterations):
            leaf = self.selection(self.root)
            if leaf.state.status != leaf.state.ONGOING:
                self.back_propagate(leaf, self.simulation(leaf))
                continue
            child = self.expand(leaf)
            self.back_propagate(child, self.simulation(child))

    def choose_move(self, game):
        self._run_search(game)
        return max(self.root.children.items(), key=lambda kv: kv[1].N)[0]

    def choose_move_with_stats(self, game):
        self._run_search(game)
        board_size = game.BOARD_SIZE
        visit_counts = np.zeros(board_size * board_size, dtype=np.float32)
        for move, child in self.root.children.items():
            row, col = move
            visit_counts[row * board_size + col] = child.N
        best_move = max(self.root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_move, visit_counts


def play_vs_mcts():
    game = Gomoku()
    ai = MCTS(game, c=1.6, iterations=4000)

    print("=== Game vs MCTS ===")
    print("You are BLACK (X). AI is WHITE (O).\n")

    while game.status == game.ONGOING:
        print(game)
        if game.player == game.BLACK:
            while True:
                try:
                    parts = input("Enter row and column: ").strip().split()
                    if len(parts) != 2:
                        raise ValueError
                    move = (int(parts[0]), int(parts[1]))
                    if move not in game.legal_moves():
                        print("Illegal move. Try again.")
                        continue
                    game.make(move)
                    break
                except ValueError:
                    print("Invalid input. Use: row col")
        else:
            print("\nAI is thinking...")
            move = ai.choose_move(game)
            print("AI played:", move)
            game.make(move)
        print("\n---------------------------\n")

    print(game)
    if game.status == game.BLACK_WIN:
        print("\nYou win!")
    elif game.status == game.WHITE_WIN:
        print("\nAI wins!")
    else:
        print("\nDraw!")


if __name__ == "__main__":
    play_vs_mcts()
