"""Generate (state, policy, value) samples from classic MCTS self-play."""

import os
import pickle
import time
from gomoku_class import Gomoku
from mcts import MCTS

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_self_play_game(mcts_player):
    game = Gomoku()
    history = []

    while game.status == game.ONGOING:
        state = game.encode()
        player = game.player
        move, visit_counts = mcts_player.choose_move_with_stats(game)
        policy_target = visit_counts / visit_counts.sum()
        history.append((state, policy_target, player))
        game.make(move)

    winner = game.winner()
    if winner == game.BLACK:
        final_value = 1
    elif winner == game.WHITE:
        final_value = -1
    else:
        final_value = 0

    training_data = []
    for state, policy, player in history:
        value = final_value if player == game.BLACK else -final_value
        training_data.append((state, policy, value))
    return training_data, game.status


def main():
    mcts_player = MCTS(Gomoku(), c=1.6, iterations=4000)
    all_data = []
    num_games = 10
    start = time.time()

    for i in range(num_games):
        game_data, _ = generate_self_play_game(mcts_player)
        all_data.extend(game_data)
        print(f"{i + 1}/{num_games} games")

    out_path = os.path.join(HERE, "data_from_MCTS.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_data, f)

    print(f"Saved {len(all_data)} samples to {out_path}")
    print(f"Total time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
