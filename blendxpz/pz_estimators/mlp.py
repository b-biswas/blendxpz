import logging

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input, PReLU, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

tfd = tfp.distributions

LOG = logging.getLogger(__name__)

def create_mpl_estimator(num_filters=5):

    input_layer = Input(shape=(num_filters))

    h = tf.keras.layers.BatchNormalization()(input_layer)

    h = Dense(64)(h)
    h = PReLU()(h)
    h = Dropout(.05)(h)

    h = Dense(128)(h)
    h = Dropout(.05)(h)
    h = PReLU()(h)

    h = tf.keras.layers.BatchNormalization()(h)

    h = Dense(128)(h)
    h = PReLU()(h)
    h = Dropout(.05)(h)

    h = Dense(64)(h)
    h = PReLU()(h)
    h = Dropout(.05)(h)

    mu = Dense(1, activation="relu")(h)
    # sig = Dense(1, activation="relu")(h) + 0.01
    return Model(input_layer, mu, name="catalog-pz-estimator")
