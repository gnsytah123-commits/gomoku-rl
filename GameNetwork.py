import tensorflow as tf


def build_game_network(board_size=7):
    num_actions = board_size * board_size

    #3 chnnels
    inputs = tf.keras.Input(shape=(board_size, board_size, 3))

    # 3 שכבות שלומדות את הלוח
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    #הפלט הוא 64 מטריצות בגודל 7

    # Policy 
    p = tf.keras.layers.Conv2D(2, 1, activation="relu")(x) #סכימה
    p = tf.keras.layers.Flatten()(p)
    policy = tf.keras.layers.Dense(num_actions, activation="softmax", name="policy")(p)

    # Value
    v = tf.keras.layers.Conv2D(1, 1, activation="relu")(x) #שקלול כל מיקום תא מה-64 מטריצות ומחזיר מטריצה אחת
    v = tf.keras.layers.Flatten()(v)
    v = tf.keras.layers.Dense(64, activation="relu")(v) #עוד שכבה של למידה כי קשה לסכם לוח שלם לספרה אחת, מחזיר וקטור באורך 64
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(v) #בסוף הפלט הוא מספר אחד

    model = tf.keras.Model(inputs=inputs, outputs=[policy, value])
    return model


