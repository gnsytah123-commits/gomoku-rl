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
        
        self.N = 0          # מספר ביקורים
        self.Q = 0          # mean value for the player to move at THIS node
        self.P = prior      # policy
    
    def get_puct_score(self, c_puct):
        parent_n = self.parent.N if self.parent else 1 #בשביל השורש
        # Parent selects, so negate Q (child Q is the opponent's value).
        return -self.Q + c_puct * self.P * (math.sqrt(parent_n) / (1 + self.N))

    def is_expanded(self): #האם עשינו לקודקוד expand
        return len(self.children) > 0
    


class PUCTPlayer:
    def __init__(self, model, c_puct=1.5, iterations=100):
        self.model = model  # כאן ייכנס המודל מה-Pre-training
        self.c_puct = c_puct
        self.iterations = iterations

    def select_child(self, node):
        # בוחרים את הילד עם הציון הגבוה ביותר
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.get_puct_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand_and_evaluate(self, node):
        # 1. הכנת הקלט לרשת
        encoded_state = node.state.encode().transpose(1, 2, 0)
        encoded_state = np.expand_dims(encoded_state, axis=0).astype(np.float32)  # (7, 7, 3) -> (1, 7, 7, 3)
        
        policy_output, value_output = self.model(encoded_state, training=False)
        
        #חילוץ הפלט
        policy_probs = policy_output[0].numpy()
        value = value_output[0][0].numpy()
        
        # 3. טיפול במהלכים חוקיים בלבד
        possible_moves = node.state.legal_moves()
        
        #  יצירת מסכה למהלכים החוקיים שתשמש אותנו לpolicy
        # וקטור בגודל 49 כשאר יש 1 בכל מהלך חוקי
        mask = np.zeros_like(policy_probs)
        for move in possible_moves:
            move_index = move[0] * node.state.BOARD_SIZE + move[1]
            mask[move_index] = 1
        
        # במהלכים לא חוקיים יהיה 0
        policy_probs *= mask
        prob_sum = np.sum(policy_probs)
        if prob_sum > 0:
            policy_probs /= prob_sum #נרמול הוקטור
        else:
            # מקרה קצה: אם הרשת נתנה 0 לכל המהלכים החוקיים, נחלק הסתברות שווה
            policy_probs = mask / np.sum(mask)

        # 4. יצירת צמתי הילדים
        for move in possible_moves:
            move_index = move[0] * node.state.BOARD_SIZE + move[1]
            prior = policy_probs[move_index]
            
            next_game = node.state.clone()
            next_game.make(move)
            
            node.children[move] = PUCTNode(
                state=next_game, 
                parent=node, 
                move=move, 
                prior=prior
            )
        
        return value 


    def back_propagate(self, node, value):
        curr = node
        while curr is not None:
            curr.N += 1
            curr.Q += (value - curr.Q) / curr.N
            
            # הפיכת הערך עבור ההורה (השחקן הקודם)
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
        best_move = max(root.children.items(), key=lambda x: x[1].N)[0]
        return best_move

    def choose_move_with_stats(self, game, training=True):
        root = self._search(game)
        board_size = game.BOARD_SIZE
        visit_counts = np.zeros(board_size * board_size, dtype=np.float32)
        
        for move, child in root.children.items():
            idx = move[0] * board_size + move[1]
            visit_counts[idx] = child.N
            
        if training: #דגימה הסבתרותית
            probs = visit_counts / np.sum(visit_counts)
            move_index = np.random.choice(len(probs), p=probs)
        else: #טסט
            move_index = np.argmax(visit_counts)

        chosen_move = game.decode(move_index)
        return chosen_move, visit_counts



def play_against_ai(model_path):
    import tensorflow as tf

    game = Gomoku()
    model = tf.keras.models.load_model(model_path)
    ai_player = PUCTPlayer(model, c_puct=1.5, iterations=10) 

    while game.status == game.ONGOING:
        print(game)
        if game.player == game.BLACK:
            # --- תור אנושי עם מנגנון "ניסיון חוזר" ---
            while True:
                try:
                    move_input = input("Your move (row col): ").strip().split()
                    if len(move_input) != 2:
                        print("Please enter TWO numbers (e.g., '3 4')")
                        continue
                    
                    move = (int(move_input[0]), int(move_input[1]))
                    
                    # הניסיון לבצע את המהלך
                    game.make(move)
                    break  # אם הצלחנו, יוצאים מהלולאה הפנימית וממשיכים ל-AI
                
                except ValueError as e:
                    # תופס משבצת תפוסה או פורמט לא תקין
                    print(f"Invalid move: {e}. Try again.")
                except IndexError:
                    # תופס חריגה מגבולות הלוח
                    print("Move out of bounds! Must be between 0 and 6. Try again.")
                except Exception as e:
                    print(f"An error occurred: {e}")
        else:
            # תור AI
            print("AI is thinking...")
            move = ai_player.choose_move(game)
            game.make(move)
            print(f"AI played: {move}")

    print(game)
    if game.status == game.BLACK_WIN:
        print("\nYOU WIN")
    elif game.status == game.WHITE_WIN:
        print("\nYOU LOSE ")
    else:
        print("\nDRAW!")
    print("="*20)



def ai_against_ai(model_path):
    import tensorflow as tf

    game = Gomoku()
    model = tf.keras.models.load_model(model_path)
    ai_player = PUCTPlayer(model, iterations=250)

    while game.status == game.ONGOING:
        print(game)
        print("AI is thinking...")
        move = ai_player.choose_move(game)
        game.make(move)
        print(f"AI played: {move}")

    print("\n" + "="*20)
    print(game)
    print("="*20)

    if game.status == game.BLACK_WIN:
        print("Final Result: X WINNER (Black)")
    elif game.status == game.WHITE_WIN:
        print("Final Result: O WINNER (White)")
    elif game.status == game.DRAW:
        print("Final Result: IT'S A DRAW")
    else:
        print(f"Final Result: {game.status}")


if __name__ == "__main__":
    play_against_ai(os.path.join(HERE, "final_model.1000.keras"))
