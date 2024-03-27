import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def create_pz_estimator(latent_dim):

    input_layer = Input(shape=(latent_dim))
    h = Dense(32, "tanh")(input_layer)
    # h = PReLU()(h)
    h = Dense(64, "tanh")(h)
    # h = PReLU()(h)
    h = Dense(128, "tanh")(h)
    # h = PReLU()(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = Dense(64, "tanh")(h)
    # h = PReLU()(h)
    h = Dense(32, "tanh")(h)
    # h = PReLU()(h)
    mu = Dense(1, activation="relu")(h)
    # sig = Dense(1, activation="relu")(h) + 0.01
    return Model(input_layer, mu, name="pz-estimator")
