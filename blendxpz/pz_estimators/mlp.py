import logging

import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

tfd = tfp.distributions

LOG = logging.getLogger(__name__)


def create_mpl_estimator(num_filters=5):

    input_layer = Input(shape=(num_filters))

    h = Dense(128, activation="tanh")(input_layer)
    # h = Dropout(.2)(h)

    h = Dense(256, activation="tanh")(h)
    # h = Dropout(.2)(h)

    h = Dense(512, activation="tanh")(h)
    # h = Dropout(.2)(h)

    h = Dense(256, activation="tanh")(h)
    # h = Dropout(.05)(h)

    h = Dense(128, activation="tanh")(h)
    # h = Dropout(.05)(h)

    mu = Dense(1, activation="relu")(h)
    # sig = Dense(1, activation="relu")(h) + 0.01
    return Model(input_layer, mu, name="catalog-pz-estimator")
