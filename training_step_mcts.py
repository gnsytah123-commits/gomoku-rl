"""Train a new network from MCTS self-play data."""

import os
import pickle
import numpy as np
import tensorflow as tf
from GameNetwork import build_game_network

HERE = os.path.dirname(os.path.abspath(__file__))


def load_self_play_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    states = np.array([s.transpose(1, 2, 0) for (s, p, v) in data])
    policy_targets = np.array([p for (s, p, v) in data])
    value_targets = np.array([v for (s, p, v) in data], dtype=np.float32)
    return states, policy_targets, value_targets


@tf.function
def train_step(model, states, policy_targets, value_targets, optimizer):
    with tf.GradientTape() as tape:
        policy_pred, value_pred = model(states, training=True)
        policy_loss = -tf.reduce_mean(
            tf.reduce_sum(policy_targets * tf.math.log(policy_pred + 1e-8), axis=1)
        )
        value_loss = tf.reduce_mean(
            tf.square(value_targets - tf.squeeze(value_pred, axis=1))
        )
        loss = policy_loss + value_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, policy_loss, value_loss


def main():
    batch_size = 128
    epochs = 25
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = build_game_network(board_size=7)

    data_path = os.path.join(HERE, "data_from_MCTS.pkl")
    states, policy_targets, value_targets = load_self_play_data(data_path)
    dataset = tf.data.Dataset.from_tensor_slices(
        (states, policy_targets, value_targets)
    ).shuffle(len(states)).batch(batch_size)

    print("Starting training...")
    for epoch in range(epochs):
        losses, p_losses, v_losses = [], [], []
        for s_batch, p_batch, v_batch in dataset:
            loss, p_loss, v_loss = train_step(
                model, s_batch, p_batch, v_batch, optimizer
            )
            losses.append(loss.numpy())
            p_losses.append(p_loss.numpy())
            v_losses.append(v_loss.numpy())
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={np.mean(losses):.4f} | "
            f"policy={np.mean(p_losses):.4f} | "
            f"value={np.mean(v_losses):.4f}"
        )

    out_path = os.path.join(HERE, "1st_model.keras")
    model.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
