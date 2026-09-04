# תפקיד המחלקה הוא לאמן את הרשת ולהוסיף עליה דאטה של puct
# כמובן ששומרים את המודל הישן שאומן על MCTS
import tensorflow as tf
import pickle
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

def load_self_play_data(path):
    print(f"Loading data from {path}...")
    with open(path, "rb") as f:
        data = pickle.load(f)

    states = []
    for (s, p, v) in data:
        s_transposed = s.transpose(1, 2, 0)
        states.append(s_transposed)
    
    states = np.array(states)
    policy_targets = np.array([p for (s, p, v) in data])
    value_targets = np.array([v for (s, p, v) in data], dtype=np.float32)

    print(f"Loaded {len(states)} states with shape {states.shape[1:]}")
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
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #טעינת המודל הישן שעליו נוסיף דאטה חדש
    old_model_path = os.path.join(HERE, "pretrained_7x7c.keras")

    model = tf.keras.models.load_model(old_model_path)
    print(f"Model loaded from {old_model_path}. Starting incremental training...")

    #load data of puct
    data_path = os.path.join(HERE, "trained_7x7.1000.pkl")

    states, policy_targets, value_targets = load_self_play_data(data_path)

    # יצירת Dataset של Tensorflow להזרמת הנתונים
    dataset = tf.data.Dataset.from_tensor_slices(
        (states, policy_targets, value_targets)
    ).shuffle(len(states)).batch(batch_size)

    # 3. לולאת האימון
    print("Starting training session...")
    for epoch in range(epochs):
        losses, p_losses, v_losses = [], [], []

        for s_batch, p_batch, v_batch in dataset:
            loss, p_loss, v_loss = train_step(model, s_batch, p_batch, v_batch, optimizer)
            losses.append(loss.numpy())
            p_losses.append(p_loss.numpy())
            v_losses.append(v_loss.numpy())

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Total Loss={np.mean(losses):.4f} | "
            f"Policy Loss={np.mean(p_losses):.4f} | "
            f"Value Loss={np.mean(v_losses):.4f}"
        )

    
    new_model_path = os.path.join(HERE, "final_model.100b.keras")
    model.save(new_model_path)
    
    print("-" * 30)
    print(f"Training completed!")
    print(f"Original model remains at: {old_model_path}")
    print(f"Improved model saved to: {new_model_path}")

if __name__ == "__main__":
    main()