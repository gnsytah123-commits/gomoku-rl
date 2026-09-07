"""Round-robin Elo helper. Default match: 2nd_model vs 3rd_model."""

import os
import tensorflow as tf
from gomoku_class import Gomoku
from PUCT import PUCTPlayer

HERE = os.path.dirname(os.path.abspath(__file__))


def get_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating, actual_score, expected_score, k=32):
    return rating + k * (actual_score - expected_score)


def run_tournament(player1_obj, player2_obj, games=10):
    r1, r2 = 1000, 1000
    stats = {"P1_wins": 0, "P2_wins": 0, "Draws": 0}

    for i in range(games):
        game = Gomoku()
        p1_is_black = i % 2 == 0

        while game.status == game.ONGOING:
            p1_to_move = (game.player == game.BLACK and p1_is_black) or (
                game.player == game.WHITE and not p1_is_black
            )
            move = player1_obj.choose_move(game) if p1_to_move else player2_obj.choose_move(game)
            game.make(move)

        winner = game.winner()
        if winner is None:
            score1, score2 = 0.5, 0.5
            stats["Draws"] += 1
        elif (winner == game.BLACK and p1_is_black) or (
            winner == game.WHITE and not p1_is_black
        ):
            score1, score2 = 1, 0
            stats["P1_wins"] += 1
        else:
            score1, score2 = 0, 1
            stats["P2_wins"] += 1

        r1 = update_elo(r1, score1, get_expected_score(r1, r2))
        r2 = update_elo(r2, score2, get_expected_score(r2, r1))
        print(
            f"Game {i + 1}/{games}  "
            f"P1 {stats['P1_wins']} - P2 {stats['P2_wins']} - D {stats['Draws']}  "
            f"(P2 Elo {r2:.1f})"
        )

    return stats, r1, r2


if __name__ == "__main__":
    model_2 = tf.keras.models.load_model(os.path.join(HERE, "2nd_model.keras"))
    model_3 = tf.keras.models.load_model(os.path.join(HERE, "3rd_model.keras"))
    player1 = PUCTPlayer(model_2, c_puct=5.0, iterations=1000)
    player2 = PUCTPlayer(model_3, c_puct=5.0, iterations=1000)

    num_games = 40
    print(f"P1=2nd_model  P2=3rd_model  ({num_games} games, colors swapped)")
    stats, elo1, elo2 = run_tournament(player1, player2, games=num_games)

    print("-" * 60)
    print(f"P1 (2nd) wins: {stats['P1_wins']}")
    print(f"P2 (3rd) wins: {stats['P2_wins']}")
    print(f"Draws:         {stats['Draws']}")
    print(f"P1 Elo:        {elo1:.1f}")
    print(f"P2 Elo:        {elo2:.1f}")
