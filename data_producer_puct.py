#תפקיד המחלקה לייצר דאטה של Puct
import numpy as np
import os
import pickle
import time
import tensorflow as tf
from gomoku_class import Gomoku
from PUCT import PUCTPlayer

HERE = os.path.dirname(os.path.abspath(__file__))

def generate_self_play_game_puct(puct_player):
    game = Gomoku()
    history = [] # ישמור (state, policy_target, current_player)

    while game.status == game.ONGOING:
        # שימוש בדגימה הסתברותית (training=True) כפי שנדרש לייצור דאטה
        move, visit_counts = puct_player.choose_move_with_stats(game, training=True)
        
        # הכנת ה-Policy Target על ידי נרמול הביקורים
        policy_target = visit_counts / np.sum(visit_counts)
        
        # שמירת המצב (encode), המטרה (policy) והשחקן הנוכחי
        history.append((game.encode(), policy_target, game.player))
        
        game.make(move)

    # קביעת ה-Value הסופי (1 לניצחון שחור, -1 ללבן, 0 לתיקו)
    if game.status == game.BLACK_WIN:
        final_result = 1
    elif game.status == game.WHITE_WIN:
        final_result = -1
    else:
        final_result = 0

    # התאמת הנתונים לאימון (Value מנקודת המבט של השחקן ששיחק)
    training_data = []
    for state, policy, player in history:
        value = final_result if player == game.BLACK else -final_result
        training_data.append((state, policy, value))

    return training_data

def run_data_generation(model_path, num_games=1000, iterations=1000):
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    puct_player = PUCTPlayer(model, c_puct=0.8, iterations=iterations)
    
    all_data = []
    start_time = time.time()  # שמירת זמן התחלה

    for i in range(num_games):
        game_data = generate_self_play_game_puct(puct_player)
        all_data.extend(game_data)
        
        # הדפסת התקדמות מינימלית בלי חישוב זמן
        if (i + 1) % 10 == 0:
            print(f"Game {i+1}/{num_games} finished. Total samples: {len(all_data)}")

    output_file = os.path.join(HERE, "trained_7x7.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(all_data, f)
    
    # חישוב והדפסת זמן הריצה הכולל רק בסיום
    end_time = time.time()
    total_duration = (end_time - start_time) / 60  # המרה לדקות
    
    print("-" * 30)
    print(f"Generation completed.")
    print(f"Saved to: {output_file}")
    print(f"Total execution time: {total_duration:.2f} minutes")
    print("-" * 30)

if __name__ == "__main__":
    run_data_generation(os.path.join(HERE, "pretrained_7x7c.keras"), num_games=1000, iterations=1000)