import tensorflow as tf


def create_inner_model(layer_sizes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization())

    for layer_size in layer_sizes:
        dense = tf.keras.layers.Dense(
            units=layer_size,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.lecun_normal(),
            bias_initializer=tf.keras.initializers.lecun_normal()
        )
        model.add(dense)

    dense = tf.keras.layers.Dense(
        units=2,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        bias_initializer=tf.keras.initializers.lecun_normal()
    )

    model.add(dense)

    return model
