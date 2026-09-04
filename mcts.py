from gomoku_class import *
import time

class Node:
    def __init__(self, state, parent=None, move=None): #state מסוג Gomoku
        self.state = state #  clone - עותק של מצב המשחק
        self.parent = parent    
        self.move = move  # איזה מהלך הוביל למצב הזה

        self.children = {} # רשימה של nodes
        self.N = 0 #מספר הביקורים בצומת
        self.W = 0 #סכום התוצאות שעברו דרך הצומת rollout
        self.Q = 0 # ממוצע


import math
import random
import numpy as np


class MCTS:
    def __init__(self, game, c = 0.8, iterations = 2000): #מקבל אובייקט Gomoku
        self.root = Node(game.clone()) #יוצר נוד שהוא עותק של מצב המשחק וממנו נחשב את העץ 
        self.c = c
        self.root_player = game.player #נקודת מבט של איזה שחקן
        self.iterations = iterations

    # מקבל צומת ומחשבת לה את UCT - uct
    def UCT(self, node):
        if node.N == 0: #אם לא בקרנו בצומת נבחר אותו
            return float('inf') #נרצה לבקר בו
        
        if node.parent is None or node.parent.N == 0: #בדיקת בטיחות
            return float('inf') 

        return node.Q + self.c * math.sqrt(math.log(node.parent.N) / node.N)

    #מהשורש עד שמגיעים לעלה 
    def selection(self, node):
        cur = node
        while cur.children and len(cur.children) == len(cur.state.legal_moves()) and cur.state.status == cur.state.ONGOING:
            
            best_child = max(cur.children.values(), key=lambda c: self.UCT(c))
            
            cur = best_child
        return cur
       
    
    #expand
    #אם הצומת שהגענו אליו לא מצב סיום ואין לו את כל הילדים האפשריים
    #ניצור ילד חדש בעץ לפי אחד המהלכים החוקיים
    #נוסיף אותו לchildren של הצומת
    #נחזיר את הצומת החדש כדי לעשות עליו rollout
    def expand(self, node):
        legal_moves = node.state.legal_moves() #חישוב המהלכים החוקיים

        unexpanded_moves = [] #מהלכים שעוד לא יצרנו עבורם צומת בעץ
        
        for move in legal_moves:
            if move not in node.children:
                unexpanded_moves.append(move)

        if not unexpanded_moves: #אם אין כאלו
            return node

        move = random.choice(unexpanded_moves) #נבחר מהלך רנדומלי להרחבה

        new_state = node.state.clone() #משכפל את מצב המשחק
        new_state.make(move) #נבצע מהלך

        child = Node(new_state, parent=node, move=move) #ניצור את הצומת החדש
        node.children[move] = child #נוסיף את הצומת לבנים של האבא

        return child
    

    #simulation
    #מקבל צומת חדש, משחק במהלכים אקראיים עד סוף המשחק
    def simulation(self, node):
        state = node.state.clone()
        while state.status == state.ONGOING:
            move = random.choice(state.legal_moves())
            state.make(move)

        winner = state.winner()
        if winner is None:
            return 0

        # Reward from the player who moved TO this node (parent). That is
        # whose Q the parent maximizes when choosing among children.
        if node.parent is None:
            mover = state.other(state.player)
        else:
            mover = node.parent.state.player
        return 1 if winner == mover else -1


    #back propogation
    #קבלנו תוצאה כלשהי בsimulation , נרצה לעלות במעלה העץ בצמתים שעברנו ולעדכן את N, Q, W
    def back_propagate(self, node, reward):
        cur = node
        r = reward # הפרס שחזר מהסימולציה (ביחס למי ששיחק אחרון)
        
        while cur is not None:
            cur.N += 1
            cur.W += r
            cur.Q = cur.W / cur.N

            r = -r 
            cur = cur.parent


    #נבנה עץ מהעמדה שקיבל כאשר היא השורש ומריצים משם משחק
    def choose_move(self, game): #להריץ mcts ולהחזיר את המהלך הטוב ביותר מבין המלכים האפשרים באותה עמדה

        self.root = Node(game.clone())     # שורש חדש לכל הרצה
        self.root_player = game.player     # קביעת נק המבט

        for _ in range(self.iterations):
            
            # selection - מחזיר עלה
            leaf = self.selection(self.root)

            # אם זה מצב סיום משחק
            if leaf.state.status != leaf.state.ONGOING:
                reward = self.simulation(leaf)  # נייבא את התוצאה
                self.back_propagate(leaf, reward) #פעפוע התוצאה
                continue

            # אם הגענו לעלה שהוא לא מצב סיום המשחק
            #יצירת קודקוד חדש
            child = self.expand(leaf)

            # simulation
            reward = self.simulation(child)

            # backprop
            self.back_propagate(child, reward)

        # לבחור את הילד עם N הכי גבוה
        best_move = max(self.root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_move
    
    #מתאים לאלפא זירו ששם משתמשים בהתפלגות הביקורים של המהלכים בלוח
    def choose_move_with_stats(self, game):
        # בונים עץ חדש כמו ב-choose_move
        self.root = Node(game.clone())
        self.root_player = game.player

        for _ in range(self.iterations):
            leaf = self.selection(self.root)

            if leaf.state.status != leaf.state.ONGOING:
                reward = self.simulation(leaf)
                self.back_propagate(leaf, reward)
                continue

            child = self.expand(leaf)
            reward = self.simulation(child)
            self.back_propagate(child, reward)

        #visit counts
        board_size = game.BOARD_SIZE
        visit_counts = np.zeros(board_size * board_size, dtype=np.float32)

        for move, child in self.root.children.items():
            row, col = move
            action_index = row * board_size + col
            visit_counts[action_index] = child.N

        
        best_move = max(self.root.children.items(), key=lambda kv: kv[1].N)[0]

        return best_move, visit_counts


def play_vs_mcts():
    game = Gomoku()
    ai = MCTS(game, c=1.6, iterations=4000)

    print("=== Starting a game vs MCTS ===")
    print("You are BLACK (X)")
    print("AI is WHITE (O)\n")

    while game.status == game.ONGOING:
        print(game)
        print("\nYour turn! Choose a row and column (0-6):")

        # --- Human turn ---
        if game.player == game.BLACK:
            while True:
                try:
                    move_input = input("Enter row and column: ").strip().split()
                    if len(move_input) != 2:
                        raise ValueError

                    move = tuple(map(int, move_input))

                    if move not in game.legal_moves():
                        print("Illegal move. Try again.")
                        continue
                    break
                except:
                    print("Invalid input. Please enter again.")
            game.make(move)

        # --- AI turn ---
        else:
            print("\nAI is thinking...")
            ai_move = ai.choose_move(game)
            print("AI played:", ai_move)
            game.make(ai_move)

        print("\n---------------------------\n")

    # --- End of game ---
    print(game)

    if game.status == game.BLACK_WIN:
        print("\nYou win!")
    elif game.status == game.WHITE_WIN:
        print("\nAI wins!")
    else:
        print("\nDraw!")

def ai_vs_himself():
    game = Gomoku()
    ai = MCTS(game, iterations=2000)
    while game.status == game.ONGOING:
        print(game)

        print("\nAI is thinking...")
        ai_move = ai.choose_move(game)
        game.make(ai_move)

        print("\n---------------------------\n")

    # --- End of game ---
    print(game)

    if game.status == game.BLACK_WIN:
        print("\nBLACK win!")
    elif game.status == game.WHITE_WIN:
        print("\nWHITE wins!")
    else:
        print("\nDraw!")

if __name__ == "__main__":
    play_vs_mcts()
    '''start = time.time()
    ai_vs_himself()
    end = time.time()
    print(f"{end - start} seconds")'''