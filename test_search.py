import unittest
import numpy as np

from gomoku_class import Gomoku
from PUCT import PUCTNode, PUCTPlayer


def four_in_a_row_then_white():
    """Black has 4 on row 3; White to have just played. Black to move, (3,4) wins."""
    game = Gomoku()
    for move in [
        (3, 0), (0, 0),
        (3, 1), (0, 1),
        (3, 2), (0, 2),
        (3, 3), (0, 3),
    ]:
        game.make(move)
    return game


class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def numpy(self):
        return self._arr

    def __getitem__(self, i):
        item = self._arr[i]
        return FakeTensor(item)


class UniformModel:
    """Policy uniform over 49 squares, value 0. Duck-types a Keras model call."""

    def __call__(self, x, training=False):
        policy = np.ones((1, 49), dtype=np.float32) / 49.0
        value = np.array([[0.0]], dtype=np.float32)
        return FakeTensor(policy), FakeTensor(value)


class GomokuTests(unittest.TestCase):
    def test_five_in_a_row_and_player_switch(self):
        game = four_in_a_row_then_white()
        self.assertEqual(game.status, game.ONGOING)
        self.assertEqual(game.player, game.BLACK)

        game.make((3, 4))
        self.assertEqual(game.status, game.BLACK_WIN)
        self.assertEqual(game.winner(), game.BLACK)
        self.assertNotEqual(game.BLACK_WIN, game.BLACK)
        # To-move after a win is the loser, so backups use -1.
        self.assertEqual(game.player, game.WHITE)
        self.assertEqual(game.terminal_value(), -1.0)

    def test_four_in_a_row_does_not_win(self):
        game = four_in_a_row_then_white()
        self.assertEqual(game.status, game.ONGOING)
        self.assertIsNone(game.winner())

    def test_unmake_after_win_restores_side_to_move(self):
        game = four_in_a_row_then_white()
        game.make((3, 4))
        game.unmake((3, 4))
        self.assertEqual(game.status, game.ONGOING)
        self.assertEqual(game.player, game.BLACK)
        self.assertEqual(game.board[3][4], game.EMPTY)

    def test_diagonal_win(self):
        game = Gomoku()
        for move in [
            (0, 0), (0, 1),
            (1, 1), (0, 2),
            (2, 2), (0, 3),
            (3, 3), (0, 4),
            (4, 4),
        ]:
            game.make(move)
        self.assertEqual(game.status, game.BLACK_WIN)
        self.assertEqual(game.player, game.WHITE)


class PuctScoreTests(unittest.TestCase):
    def test_parent_prefers_child_where_opponent_lost(self):
        parent = PUCTNode(Gomoku())
        parent.N = 10
        winning_child = PUCTNode(Gomoku(), parent=parent, move=(3, 4), prior=0.1)
        winning_child.Q = -1.0  # to-move at child lost
        winning_child.N = 1
        other = PUCTNode(Gomoku(), parent=parent, move=(1, 1), prior=0.1)
        other.Q = 0.0
        other.N = 1
        self.assertGreater(
            winning_child.get_puct_score(1.5),
            other.get_puct_score(1.5),
        )


class PuctSearchTests(unittest.TestCase):
    def test_picks_immediate_win(self):
        game = four_in_a_row_then_white()
        player = PUCTPlayer(UniformModel(), c_puct=1.5, iterations=64)
        move = player.choose_move(game)
        self.assertEqual(move, (3, 4))


class MctsBackupTests(unittest.TestCase):
    def test_terminal_reward_is_from_the_mover(self):
        from mcts import MCTS, Node

        parent_game = four_in_a_row_then_white()
        parent = Node(parent_game)
        child_game = parent_game.clone()
        child_game.make((3, 4))
        child = Node(child_game, parent=parent, move=(3, 4))

        self.assertEqual(child_game.status, child_game.BLACK_WIN)
        self.assertEqual(child_game.player, child_game.WHITE)

        ai = MCTS(parent_game, iterations=1)
        self.assertEqual(ai.simulation(child), 1)


if __name__ == "__main__":
    unittest.main()
