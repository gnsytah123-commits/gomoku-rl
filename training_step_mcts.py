#תפקיד המחלקה לאמן את הרשת על דאטה של MCTS
import tensorflow as tf
from GameNetwork import build_game_network
import pickle
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_self_play_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    states = np.array([s.transpose(1, 2, 0) for (s, p, v) in data]) #לוקח את S והופך את הסדר
    policy_targets = np.array([p for (s, p, v) in data])
    value_targets = np.array([v for (s, p, v) in data], dtype=np.float32)

    return states, policy_targets, value_targets


@tf.function
def train_step(model, states, policy_targets, value_targets, optimizer):
    with tf.GradientTape() as tape: #הקלטה של כל הפעולות המתמטיות
        policy_pred, value_pred = model(states, training=True) #הפעלת המודל על states

        policy_loss = -tf.reduce_mean( #cross entropy
            tf.reduce_sum(
                policy_targets * tf.math.log(policy_pred + 1e-8),
                axis=1
            )
        )

        value_loss = tf.reduce_mean(
            tf.square(value_targets - tf.squeeze(value_pred, axis=1))
        )

        loss = policy_loss + value_loss

    grads = tape.gradient(loss, model.trainable_variables) #חישוב הנגזרות ועדכון המשקולות
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) #עדכון המשקולות בפועל

    return loss, policy_loss, value_loss


def main():
    batch_size = 128
    epochs = 25
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model = build_game_network(board_size=7)

    data_path = os.path.join(HERE, "self_play4_augmented.pkl")
    states, policy_targets, value_targets = load_self_play_data(data_path)

    dataset = tf.data.Dataset.from_tensor_slices(
        (states, policy_targets, value_targets)
    ).shuffle(len(states)).batch(batch_size)

    print("Starting training...")

    for epoch in range(epochs):
        losses = []
        p_losses = []
        v_losses = []

        for s_batch, p_batch, v_batch in dataset:
            loss, p_loss, v_loss = train_step(model, s_batch, p_batch, v_batch, optimizer)
            losses.append(loss.numpy())
            p_losses.append(p_loss.numpy())
            v_losses.append(v_loss.numpy())

        print(
            f"Epoch {epoch}: "
            f"loss={np.mean(losses):.4f} | "
            f"policy={np.mean(p_losses):.4f} | "
            f"value={np.mean(v_losses):.4f}"
        )

        out_path = os.path.join(HERE, "pretrained_7x7e.keras")
        model.save(out_path)
        print(f"Model updated and saved to: {out_path}")

    print("Training session completed.")


if __name__ == "__main__":
    main()
