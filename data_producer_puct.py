"""Generate (state, policy, value) samples from PUCT self-play."""

import os
import pickle
import time
import numpy as np
import tensorflow as tf
from gomoku_class import Gomoku
from PUCT import PUCTPlayer

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_self_play_game_puct(puct_player):
    game = Gomoku()
    history = []

    while game.status == game.ONGOING:
        move, visit_counts = puct_player.choose_move_with_stats(game, training=True)
        policy_target = visit_counts / np.sum(visit_counts)
        history.append((game.encode(), policy_target, game.player))
        game.make(move)

    if game.status == game.BLACK_WIN:
        final_result = 1
    elif game.status == game.WHITE_WIN:
        final_result = -1
    else:
        final_result = 0

    training_data = []
    for state, policy, player in history:
        value = final_result if player == game.BLACK else -final_result
        training_data.append((state, policy, value))
    return training_data


def run_data_generation(model_path, num_games=1000, iterations=1000, c_puct=5.0):
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    puct_player = PUCTPlayer(model, c_puct=c_puct, iterations=iterations)

    all_data = []
    start = time.time()
    for i in range(num_games):
        all_data.extend(generate_self_play_game_puct(puct_player))
        if (i + 1) % 10 == 0:
            print(f"Game {i + 1}/{num_games}. Samples: {len(all_data)}")

    output_file = os.path.join(HERE, "data_from_2nd_model.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(all_data, f)

    print(f"Saved to {output_file}")
    print(f"Total time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    run_data_generation(
        os.path.join(HERE, "2nd_model.keras"),
        num_games=1000,
        iterations=1000,
        c_puct=5.0,
    )
