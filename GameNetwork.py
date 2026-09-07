import tensorflow as tf


def build_game_network(board_size=7):
    """Dual-head conv net: policy over 49 squares and a scalar value in [-1, 1]."""
    num_actions = board_size * board_size
    inputs = tf.keras.Input(shape=(board_size, board_size, 3))

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    p = tf.keras.layers.Conv2D(2, 1, activation="relu")(x)
    p = tf.keras.layers.Flatten()(p)
    policy = tf.keras.layers.Dense(num_actions, activation="softmax", name="policy")(p)

    v = tf.keras.layers.Conv2D(1, 1, activation="relu")(x)
    v = tf.keras.layers.Flatten()(v)
    v = tf.keras.layers.Dense(64, activation="relu")(v)
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(v)

    return tf.keras.Model(inputs=inputs, outputs=[policy, value])
