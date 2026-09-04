from gomoku_class import *
from PUCT import *
from mcts import *
import os
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))


def run_tournament(player1_obj, player2_obj, games=10):
    r1, r2 = 1000, 1000
    stats = {"P1_wins": 0, "P2_wins": 0, "Draws": 0}
    
    for i in range(games):
        game = Gomoku()
        p1_is_black = (i % 2 == 0)
        
        while game.status == game.ONGOING:
            if (game.player == game.BLACK and p1_is_black) or \
               (game.player == game.WHITE and not p1_is_black):
                move = player1_obj.choose_move(game)
            else:
                move = player2_obj.choose_move(game)
            game.make(move)
        
        winner = game.winner()
        if winner is None:
            score1, score2 = 0.5, 0.5
            stats["Draws"] += 1
        elif (winner == game.BLACK and p1_is_black) or \
             (winner == game.WHITE and not p1_is_black):
            score1, score2 = 1, 0
            stats["P1_wins"] += 1
        else:
            score1, score2 = 0, 1
            stats["P2_wins"] += 1
            
        exp1 = get_expected_score(r1, r2)
        exp2 = get_expected_score(r2, r1)
        r1 = update_elo(r1, score1, exp1)
        r2 = update_elo(r2, score2, exp2)
    
    return stats, r1, r2


def get_expected_score(rating_a, rating_b):
    """חישוב הסתברות הניצחון של שחקן א' מול שחקן ב'"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, actual_score, expected_score, k=32):
    """עדכון הדירוג לאחר משחק (K הוא מקדם השינוי)"""
    return rating + k * (actual_score - expected_score)    




if __name__ == "__main__":
    model_path = os.path.join(HERE, "pretrained_7x7c.keras")
    model = tf.keras.models.load_model(model_path)
    
    base_game = Gomoku()
    mcts_player = MCTS(base_game, c=1.6, iterations=4000)

    c_values = [1.5]    
    results_table = []

    print(f"{'C_PUCT':<10} | {'P1 (MCTS) Wins':<15} | {'P2 (PUCT) Wins':<15} | {'Draws':<10} | {'Final PUCT ELO':<15}")
    print("-" * 75)

    for c_val in c_values:
        # יצירת שחקן PUCT עם ה-C הנוכחי ו-1000 איטרציות
        puct_player = PUCTPlayer(model, c_puct=c_val, iterations=1000)
        
        # הרצת הטורניר
        stats, mcts_elo, puct_elo = run_tournament(mcts_player, puct_player, games=10)
        
        # הדפסת תוצאה לשורה
        print(f"{c_val:<10} | {stats['P1_wins']:<15} | {stats['P2_wins']:<15} | {stats['Draws']:<10} | {puct_elo:<15.2f}")
        
        results_table.append({
            "c": c_val,
            "stats": stats,
            "puct_elo": puct_elo
        })

    # סיכום קצר
    best_c = max(results_table, key=lambda x: x['puct_elo'])
    print(f"\nBest performing C for PUCT: {best_c['c']} with ELO {best_c['puct_elo']:.2f}")

    

'''
                                summary
mcts best on 4000 iterations and c = 1.6                                

C_PUCT | iterations   | MCTS Wins  |  PUCT Wins on 100 iterations  | Draws      | Final PUCT ELO 
-------------------------------------------------------------------------------------------------
50     | 100          | 2          | 2                             | 6          | 1011.17
                                
C_PUCT | iterations    | puct final model WINS  | puct pretrained model WINS  | Draws      | Final PUCT ELO 
------------------------------------------------------------------------------------------------------------
50     | 100           | 5                      | 5                           | 0          | 1005.19         
'''    