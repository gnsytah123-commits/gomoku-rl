"""7x7 Gomoku (five in a row). Encoding is channels-first (3, 7, 7)."""

import numpy as np


class Gomoku:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    # Distinct from BLACK/WHITE so status is never compared to a player id by accident.
    BLACK_WIN = 2
    WHITE_WIN = -2
    DRAW = 0
    ONGOING = 99

    BOARD_SIZE = 7


    def __init__(self):
        self.board = [[self.EMPTY for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.player = self.BLACK
        self.status = self.ONGOING
        self.move_count = 0

    def legal_moves(self):
        legal_moves = []
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.board[i][j] == self.EMPTY:
                    legal_moves.append((i, j))
        return legal_moves            

    def make(self, move):
        row, col = move
        if self.status != self.ONGOING:
            raise ValueError("Game already finished")

        if self.board[row][col] != self.EMPTY:
            raise ValueError("Illegal move")
        
        self.board[row][col] = self.player
        self.move_count += 1
        
        if self.winning_move(move):
            self.status = self.BLACK_WIN if self.player == self.BLACK else self.WHITE_WIN
        elif self.move_count == self.BOARD_SIZE * self.BOARD_SIZE:
            self.status = self.DRAW

        # Always switch. After a win the player-to-move is the loser, so
        # search backups share one convention: value is from the side to move.
        self.player = self.other(self.player)

    def other(self, player):
        return self.WHITE if player == self.BLACK else self.BLACK 

    def winner(self):
        if self.status == self.BLACK_WIN:
            return self.BLACK
        if self.status == self.WHITE_WIN:
            return self.WHITE
        return None

    def terminal_value(self):
        """Value from the player-to-move. After make() that is the loser, so a win is -1."""
        if self.status == self.ONGOING:
            raise ValueError("Game is not over")
        if self.status == self.DRAW:
            return 0.0
        return -1.0


    def unmake(self, move):
        row, col = move
        self.board[row][col] = self.EMPTY
        self.move_count -= 1
        self.status = self.ONGOING
        self.player = self.other(self.player)

    def clone(self):
        clone = Gomoku()
        clone.board = [row[:] for row in self.board]  
        clone.player = self.player
        clone.status = self.status
        clone.move_count = self.move_count  

        return clone

    def winning_move(self, move):
        row, col = move
        
        player = self.board[row][col] 

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 0
            x, y = row + dx, col + dy
            while 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE and self.board[x][y] == player:
                count += 1
                x += dx
                y += dy
            x, y = row - dx, col - dy
            while 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE and self.board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            if count >= 4:
                return True
        return False

    def __str__(self):
        header = "    " + " ".join([str(i) for i in range(self.BOARD_SIZE)])
        rows = [header]
        
        for r in range(self.BOARD_SIZE):
            row_chars = []
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == self.BLACK:
                    row_chars.append("X")
                elif self.board[r][c] == self.WHITE:
                    row_chars.append("O")
                else:
                    row_chars.append(".")
            
            rows.append(f"{r:2}  {' '.join(row_chars)}")
            
        return "\n".join(rows)
    

    def encode(self):
        """
        Encode the current game state as a tensor of shape (3, 7, 7)

        Channel 0: stones of the current player
        Channel 1: stones of the opponent
        Channel 2: current player indicator (all 1s or all 0s)
        """

        encoded = np.zeros((3, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)

        current_player = self.player
        opponent = self.BLACK if current_player == self.WHITE else self.WHITE

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == current_player:
                    encoded[0, r, c] = 1.0
                elif self.board[r][c] == opponent:
                    encoded[1, r, c] = 1.0

        encoded[2, :, :] = 1.0 if current_player == self.BLACK else 0.0

        return encoded
    
    def decode(self, action_index):
        """Decode an action index (0–48) into a (row, col) move."""

        if action_index < 0 or action_index >= self.BOARD_SIZE * self.BOARD_SIZE:
            raise ValueError("Invalid action index")

        row = action_index // self.BOARD_SIZE
        col = action_index % self.BOARD_SIZE

        return (row, col)


def main():
    """
    Simple main function for debugging purposes.
    Allows two human players to play Gomoku in the terminal.
    """
    game = Gomoku()

    print("Welcome to Gomoku!")
    print("Player 1 is BLACK (X) and Player 2 is WHITE (O).\n")
    print("Enter moves as: row col (both between 0 and 6)\n")

    while game.status == game.ONGOING:
        print(game)
        print("\nCurrent Player:", "BLACK" if game.player == game.BLACK else "WHITE")

        try:
            move_input = input("Enter row and column: ").strip().split()
            if len(move_input) != 2:
                raise ValueError

            move = tuple(map(int, move_input))

            game.make(move)

        except ValueError:
            print("Invalid move. Please enter: row col (0-6)")
        except IndexError:
            print("Move out of bounds. Try again.")

    print(game)

    if game.status == game.BLACK_WIN:
        print("\nBLACK wins!")
    elif game.status == game.WHITE_WIN:
        print("\nWHITE wins!")
    else:
        print("\nIt's a draw!")


if __name__ == "__main__":
    main()

