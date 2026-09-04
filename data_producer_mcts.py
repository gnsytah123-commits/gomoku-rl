#תפקיד המחלקה להריץ משחקים MCTS ולייצר דאטה שיכנס לרשת נוירונים

from gomoku_class import Gomoku
from mcts import *
import pickle
import time

def generate_self_play_game(mcts_player):   #משחק שלם
    game = Gomoku()
    history = [] #שומר את כל זיכרון העבודה של משחק בודד

    while game.status == game.ONGOING:
        state = game.encode()
        player = game.player

        move, visit_counts = mcts_player.choose_move_with_stats(game)
        policy_target = visit_counts / visit_counts.sum() #נרמול

        history.append((state, policy_target, player))
        game.make(move)

    winner = game.winner()
    if winner == game.BLACK:
        final_value = 1
    elif winner == game.WHITE:
        final_value = -1
    else:
        final_value = 0

    #בסוף הרשת תקבל 3:
    #1. המצב בו המחשק היה לפני התור
    #2. התפלגות המהלכים באותו התור 
    #3. עבור אותו התור, אם מי שהיה תורו ניצח אז value שלו יהיה חיובי ואחרת שלילי
    training_data = []
    for state, policy, player in history:
        value = final_value if player == game.BLACK else -final_value
        training_data.append((state, policy, value))

    return training_data, game.status


def main():
    game = Gomoku()
    mcts_player = MCTS(game, c=1.6, iterations=4000)

    all_data = []

    num_games = 10
    start_total = time.time()

    for i in range(num_games):
        game_data, winner = generate_self_play_game(mcts_player)

        if (i + 1) % 1 == 0:
            print(f" {i + 1} games passed")

        all_data.extend(game_data)

    with open("pre_tarining2.pkl", "wb") as f:
        pickle.dump(all_data, f)

    with open("pre_tarining2.pkl", "rb") as f:
        data = pickle.load(f)

    end_total = time.time()
    total_duration = end_total - start_total
    print(f"Total Time: {total_duration/60:.2f} minutes")

    print(len(data) / num_games)


if __name__ == "__main__":
    main()


