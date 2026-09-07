"""Policy-guided PUCT search. Values are from the player to move."""

import math
import os
import numpy as np
from gomoku_class import Gomoku

HERE = os.path.dirname(os.path.abspath(__file__))


class PUCTNode:
    def __init__(self, state, parent=None, move=None, prior=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = 0
        self.Q = 0  # mean value for the player to move at this node
        self.P = prior

    def get_puct_score(self, c_puct):
        parent_n = self.parent.N if self.parent else 1
        # Negate Q: child.Q is the opponent's value.
        return -self.Q + c_puct * self.P * (math.sqrt(parent_n) / (1 + self.N))

    def is_expanded(self):
        return len(self.children) > 0


class PUCTPlayer:
    def __init__(self, model, c_puct=5.0, iterations=100):
        self.model = model
        self.c_puct = c_puct
        self.iterations = iterations

    def select_child(self, node):
        best_score = -float("inf")
        best_child = None
        for child in node.children.values():
            score = child.get_puct_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand_and_evaluate(self, node):
        encoded = node.state.encode().transpose(1, 2, 0)
        encoded = np.expand_dims(encoded, axis=0).astype(np.float32)

        policy_output, value_output = self.model(encoded, training=False)
        policy_probs = policy_output[0].numpy()
        value = value_output[0][0].numpy()

        legal = node.state.legal_moves()
        mask = np.zeros_like(policy_probs)
        for move in legal:
            mask[move[0] * node.state.BOARD_SIZE + move[1]] = 1

        policy_probs *= mask
        total = np.sum(policy_probs)
        if total > 0:
            policy_probs /= total
        else:
            policy_probs = mask / np.sum(mask)

        for move in legal:
            prior = policy_probs[move[0] * node.state.BOARD_SIZE + move[1]]
            nxt = node.state.clone()
            nxt.make(move)
            node.children[move] = PUCTNode(nxt, parent=node, move=move, prior=prior)
        return value

    def back_propagate(self, node, value):
        curr = node
        while curr is not None:
            curr.N += 1
            curr.Q += (value - curr.Q) / curr.N
            value = -value
            curr = curr.parent

    def _evaluate_leaf(self, node):
        if node.state.status == node.state.ONGOING:
            return self.expand_and_evaluate(node)
        return node.state.terminal_value()

    def _search(self, game):
        root = PUCTNode(state=game.clone())
        for _ in range(self.iterations):
            node = root
            while node.is_expanded() and node.state.status == node.state.ONGOING:
                node = self.select_child(node)
            self.back_propagate(node, self._evaluate_leaf(node))
        return root

    def choose_move(self, game):
        root = self._search(game)
        return max(root.children.items(), key=lambda x: x[1].N)[0]

    def choose_move_with_stats(self, game, training=True):
        root = self._search(game)
        board_size = game.BOARD_SIZE
        visit_counts = np.zeros(board_size * board_size, dtype=np.float32)
        for move, child in root.children.items():
            visit_counts[move[0] * board_size + move[1]] = child.N

        if training:
            probs = visit_counts / np.sum(visit_counts)
            move_index = np.random.choice(len(probs), p=probs)
        else:
            move_index = np.argmax(visit_counts)
        return game.decode(move_index), visit_counts


def play_against_ai(model_path):
    import tensorflow as tf

    game = Gomoku()
    model = tf.keras.models.load_model(model_path)
    ai = PUCTPlayer(model, c_puct=5, iterations=100)

    while game.status == game.ONGOING:
        print(game)
        if game.player == game.BLACK:
            while True:
                try:
                    parts = input("Your move (row col): ").strip().split()
                    if len(parts) != 2:
                        print("Please enter two numbers, e.g. '3 4'")
                        continue
                    game.make((int(parts[0]), int(parts[1])))
                    break
                except ValueError as e:
                    print(f"Invalid move: {e}. Try again.")
                except IndexError:
                    print("Move out of bounds (0-6). Try again.")
        else:
            print("AI is thinking...")
            move = ai.choose_move(game)
            game.make(move)
            print(f"AI played: {move}")

    print(game)
    if game.status == game.BLACK_WIN:
        print("\nYOU WIN")
    elif game.status == game.WHITE_WIN:
        print("\nYOU LOSE")
    else:
        print("\nDRAW!")


if __name__ == "__main__":
    play_against_ai(os.path.join(HERE, "3rd_model.keras"))
