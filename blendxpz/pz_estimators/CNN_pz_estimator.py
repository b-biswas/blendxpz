"""Create the model for CNN pz-estimator."""

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, PReLU
from tensorflow.keras.models import Model


def create_model(
    input_shape=[45, 45, 5],
    filters=[32, 128, 256, 512],
    kernels=[5, 5, 5, 5],
    dense_layer_units=512,
):
    """Create the encoder.

    Parameters
    ----------
    input_shape: list
        shape of input tensor
    latent_dim: int
        size of the latent space
    filters: list
        filters used for the convolutional layers
    kernels: list
        kernels used for the convolutional layers
    dense_layer_units: int
            number of units in the dense layer

    Returns
    -------
    encoder: tf.keras.Model
       model that takes as input the image of a galaxy and projects it to the latent space.

    """
    # Input layer
    input_layer = Input(shape=(input_shape))
    h = input_layer
    # Define the model
    # h = BatchNormalization(name="batchnorm1")(input_layer)
    for i in range(len(filters)):
        h = Conv2D(
            filters[i],
            (kernels[i], kernels[i]),
            activation=None,
            padding="same",
            strides=(2, 2),
        )(h)
        h = PReLU()(h)

    h = Flatten()(h)
    h = Dense(dense_layer_units, activation="tanh")(h)
    # h = PReLU()(h)
    mu = Dense(
        1,
        activation="relu",
    )(h)

    return Model(input_layer, mu, name="CNN_pz_estimator")
