# Gomoku RL (7×7)

AlphaZero-style agent for **Gomoku on a 7×7 board** (five in a row). A convolutional policy/value network is first distilled from classic MCTS, then improved with PUCT self-play.

This repo is meant to be read and run. The checked-in checkpoint is enough to play; you do not need the self-play pickle files.

## Play a game

You are Black (`X`). Enter moves as `row col` (both `0`–`6`).

```bash
pip install -r requirements.txt
python PUCT.py
```

Loads `final_model.1000.keras`. The AI plays White.

Human vs classic MCTS (no neural net):

```bash
python mcts.py
```

## Pipeline

```
Gomoku env
    → MCTS teacher (UCT, random rollouts)
    → self-play data
    → train CNN  (pretrained checkpoint)
    → PUCT self-play (network + search)
    → train again  (final_model.1000.keras)
    → Elo vs MCTS
```

State encoding is `(3, 7, 7)`: current-player stones, opponent stones, Black-to-move plane. Keras uses channels-last `(7, 7, 3)`, so training and PUCT transpose with `(1, 2, 0)` at the network boundary.

## Network

`GameNetwork.py` — about 84k parameters:

- 3× `Conv2D(64, 3)` with ReLU
- **Policy head:** softmax over 49 squares
- **Value head:** `tanh` in `[-1, 1]`

Loss: policy cross-entropy on MCTS/PUCT visit distributions + MSE on the game outcome from the side to move.

## Search

| | MCTS | PUCT |
| --- | --- | --- |
| Prior | none (UCT) | network policy, illegal moves masked |
| Leaf eval | random rollout | network value |
| Default | `c=1.6`, 4000 sims | `c_puct=1.5`, 100–1000 sims |

Values are from the **player to move**. After a win the turn still switches (the loser is to-move), so a terminal value is `-1`. PUCT selects with `-Q + U`.

## Files

| File | Role |
| --- | --- |
| `gomoku_class.py` | Board, rules, `encode` |
| `mcts.py` | UCT teacher |
| `GameNetwork.py` | Dual-head CNN |
| `PUCT.py` | Policy-guided search + play loop |
| `data_producer_mcts.py` / `data_producer_puct.py` | Self-play datasets |
| `training_step_mcts.py` / `training_step_puct.py` | Training loops |
| `elo.py` | Elo tournament |
| `samples_examples.py` | Inspect a `.pkl` of samples |
| `test_search.py` | Win rule, undo, PUCT Q-sign, forced win |
| `final_model.1000.keras` | Playable checkpoint |

`training_step_puct.py` fine-tunes from `pretrained_7x7c.keras` if you keep that file locally. It is not required to play.

## Tests

```bash
python -m unittest test_search -v
```

## Retrain (optional)

Self-play dumps are gitignored (`*.pkl`). After generating them:

```bash
python data_producer_mcts.py    # writes pre_tarining2.pkl; point training_step_mcts.py at that path
python training_step_mcts.py
python data_producer_puct.py    # uses pretrained_7x7c.keras
python training_step_puct.py    # writes a new checkpoint, does not overwrite final_model.1000.keras
```

`elo.py` currently loads `pretrained_7x7c.keras` and plays 10 games vs MCTS-4000. That is too few games for a stable rating; use hundreds, with swapped colors, before comparing models.
