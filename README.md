# Gomoku RL (7×7)

AlphaZero-style agent for **Gomoku on a 7×7 board** (five in a row). A small conv net is first distilled from classic MCTS, then improved with two generations of PUCT self-play.

The playable checkpoint is `3rd_model.keras`. Self-play `.pkl` files are gitignored and not required to run the agent.

## Play

Python 3.10–3.12 required.

**Create and activate a virtual environment:**

   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

**Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**run the game:**
    ```bash
    python gomoku_gui.py
    ```

You are Black. Click a square. The AI (`3rd_model.keras`, `c_puct=5`) plays White. TensorFlow may take a few seconds to load on the first run.

Terminal alternatives:

```bash
python PUCT.py    # vs the neural agent
python mcts.py    # vs classic MCTS (no network)
```

## Results (40 games, colors swapped, `c_puct=5`)

| Matchup | Result (W–L–D) | Score |
| --- | --- | --- |
| 3rd vs 2nd | 20–0–20 | **75%** |
| 3rd vs MCTS-4000 | 17–4–19 | **66%** |
| 2nd vs MCTS-4000 | 16–11–13 | 56% |
| 2nd vs MCTS-4000 (`c_puct=1.5`) | 10–17–13 | 41% |

On 7×7, a higher `c_puct` (more trust in the policy prior) worked better than 1.5 because the value head is still noisy.

## Pipeline

```
MCTS teacher (UCT, random rollouts)
    → data_from_MCTS.pkl → 1st_model.keras
    → PUCT self-play → 2nd_model.keras
    → PUCT self-play (c_puct=5) → 3rd_model.keras
```

State encoding is `(3, 7, 7)`: current-player stones, opponent stones, Black-to-move plane. Keras uses channels-last `(7, 7, 3)`, so training and search transpose with `(1, 2, 0)`.

## Network

`GameNetwork.py` — ~84k parameters:

- 3× `Conv2D(64, 3)`, ReLU
- Policy head: softmax over 49 squares
- Value head: `tanh` in `[-1, 1]`

Loss: policy cross-entropy on visit distributions + MSE on the game outcome from the side to move.

## Search

| | MCTS | PUCT |
| --- | --- | --- |
| Prior | none (UCT) | network policy, illegal moves masked |
| Leaf eval | random rollout | network value |
| Defaults | `c=1.6`, 4000 sims | `c_puct=5`, 200–1000 sims |

Values are from the **player to move**. After a terminal move the turn still switches, so a win is `-1` at that node. PUCT selects with `-Q + U`.

## Files

| File | Role |
| --- | --- |
| `gomoku_class.py` | Board, rules, `encode` |
| `mcts.py` | UCT teacher |
| `GameNetwork.py` | Dual-head CNN |
| `PUCT.py` | Policy-guided search |
| `gomoku_gui.py` | Click-to-move GUI |
| `data_producer_mcts.py` / `data_producer_puct.py` | Self-play |
| `training_step_mcts.py` / `training_step_puct.py` | Training |
| `elo.py` | Tournaments |
| `test_search.py` | Win rule, undo, PUCT Q-sign |
| `1st_model.keras` | Distilled from MCTS |
| `2nd_model.keras` | First PUCT generation |
| `3rd_model.keras` | Best checkpoint |

## Tests

```bash
python -m unittest test_search -v
```

## Retrain (optional)

```bash
python data_producer_mcts.py      # writes data_from_MCTS.pkl
python training_step_mcts.py      # writes 1st_model.keras
python data_producer_puct.py      # uses 2nd_model.keras, writes data_from_2nd_model.pkl
python training_step_puct.py      # writes 3rd_model.keras
```
