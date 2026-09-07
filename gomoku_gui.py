"""Click-to-move GUI: human (Black) vs the PUCT agent (White)."""

import os
import threading
import tkinter as tk
from tkinter import font as tkfont

from gomoku_class import Gomoku
from PUCT import PUCTPlayer, HERE

CELL = 64
PAD = 16
STONE = 22


class GomokuGUI:
    def __init__(self, model_path=None, c_puct=5.0, iterations=200):
        if model_path is None:
            model_path = os.path.join(HERE, "3rd_model.keras")

        import tensorflow as tf

        self.game = Gomoku()
        self.busy = False
        self.last_move = None

        model = tf.keras.models.load_model(model_path)
        self.ai = PUCTPlayer(model, c_puct=c_puct, iterations=iterations)

        n = Gomoku.BOARD_SIZE
        size = PAD * 2 + CELL * n

        self.root = tk.Tk()
        self.root.title("Gomoku 7x7")
        self.root.resizable(False, False)

        self.status = tk.StringVar(value="Your turn (Black). Click a square.")
        tk.Label(
            self.root,
            textvariable=self.status,
            font=tkfont.Font(size=12),
            pady=8,
        ).pack()

        self.canvas = tk.Canvas(
            self.root, width=size, height=size, bg="#d4a574", highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        tk.Button(self.root, text="New game", command=self.new_game).pack(pady=8)

        self._draw_board()

    def _cell_box(self, row, col):
        x0 = PAD + col * CELL
        y0 = PAD + row * CELL
        return x0, y0, x0 + CELL, y0 + CELL

    def _cell_at(self, x, y):
        n = self.game.BOARD_SIZE
        col = (x - PAD) // CELL
        row = (y - PAD) // CELL
        if 0 <= row < n and 0 <= col < n:
            return int(row), int(col)
        return None

    def _draw_board(self):
        self.canvas.delete("all")
        n = self.game.BOARD_SIZE
        for r in range(n):
            for c in range(n):
                x0, y0, x1, y1 = self._cell_box(r, c)
                fill = "#e8c48a" if (r + c) % 2 == 0 else "#d4a574"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#5a3d1c")

                stone = self.game.board[r][c]
                if stone == self.game.EMPTY:
                    continue
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                color = "#111" if stone == self.game.BLACK else "#f4f4f4"
                outline = "#eee" if stone == self.game.BLACK else "#333"
                if self.last_move == (r, c):
                    outline = "#c0392b"
                self.canvas.create_oval(
                    cx - STONE, cy - STONE, cx + STONE, cy + STONE,
                    fill=color, outline=outline, width=2,
                )

        self._draw_result_banner()

    def _draw_result_banner(self):
        if self.game.status == self.game.ONGOING:
            return
        if self.game.status == self.game.BLACK_WIN:
            text = "You win"
        elif self.game.status == self.game.WHITE_WIN:
            text = "You lose"
        else:
            text = "Draw"

        w = int(self.canvas["width"])
        h = int(self.canvas["height"])
        self.canvas.create_text(
            w / 2,
            h / 2,
            text=text,
            fill="#c0392b",
            font=tkfont.Font(family="Arial", size=48, weight="bold"),
        )

    def _status_for_result(self):
        if self.game.status == self.game.BLACK_WIN:
            return "You win."
        if self.game.status == self.game.WHITE_WIN:
            return "You lose."
        if self.game.status == self.game.DRAW:
            return "Draw."
        return "Your turn (Black). Click a square."

    def on_click(self, event):
        if self.busy or self.game.status != self.game.ONGOING:
            return
        if self.game.player != self.game.BLACK:
            return

        cell = self._cell_at(event.x, event.y)
        if cell is None or cell not in self.game.legal_moves():
            return

        self.game.make(cell)
        self.last_move = cell
        self._draw_board()

        if self.game.status != self.game.ONGOING:
            self.status.set(self._status_for_result())
            return

        self.busy = True
        self.status.set("AI is thinking...")
        self.root.after(30, self._start_ai)

    def _start_ai(self):
        def work():
            move = self.ai.choose_move(self.game)
            self.root.after(0, lambda: self._apply_ai(move))

        threading.Thread(target=work, daemon=True).start()

    def _apply_ai(self, move):
        if self.game.status == self.game.ONGOING:
            self.game.make(move)
            self.last_move = move
        self._draw_board()
        self.busy = False
        self.status.set(self._status_for_result())

    def new_game(self):
        if self.busy:
            return
        self.game = Gomoku()
        self.last_move = None
        self.status.set("Your turn (Black). Click a square.")
        self._draw_board()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    GomokuGUI().run()
