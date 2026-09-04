import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def board_from_planes(state):
    """Rebuild an absolute Black/White board from to-move encoding."""
    ch_cur, ch_opp, ch_side = state[0], state[1], state[2]
    black_to_move = ch_side[0, 0] > 0.5
    board = np.zeros(ch_cur.shape, dtype=np.int8)
    if black_to_move:
        board[ch_cur > 0.5] = 1
        board[ch_opp > 0.5] = -1
    else:
        board[ch_cur > 0.5] = -1
        board[ch_opp > 0.5] = 1
    return board, black_to_move


def visualize_samples(file_path, num_samples=3):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"Total samples in file: {len(data)}")
    print("=" * 40)

    for i in range(min(num_samples, len(data))):
        state, policy, value = data[i]

        print(f"\nSAMPLE #{i+1}")
        print(
            f"Value from side-to-move: {value} "
            "(+1 they won the game, -1 they lost, 0 draw)"
        )

        print("Board State:")
        if getattr(state, "ndim", 0) == 3:
            display_board, black_to_move = board_from_planes(state)
            print("  Side to move:", "BLACK (X)" if black_to_move else "WHITE (O)")
        else:
            display_board = state

        for row in display_board:
            row_str = " ".join(
                ["X" if cell == 1 else "O" if cell == -1 else "." for cell in row]
            )
            print(f"  {row_str}")

        top_indices = np.argsort(policy)[-5:][::-1]
        print("\nTop 5 moves (index, probability):")
        for idx in top_indices:
            prob = policy[idx]
            if prob > 0:
                print(f"  Move index {idx}: {prob:.2%}")

        print("-" * 40)


if __name__ == "__main__":
    visualize_samples(os.path.join(HERE, "trained_7x7.1000.pkl"), num_samples=3)
