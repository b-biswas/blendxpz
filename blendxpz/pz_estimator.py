import logging

import tensorflow as tf
import tensorflow_probability as tfp

from blendxpz.model import create_pz_estimator

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from blendxpz.model import create_pz_estimator

tfd = tfp.distributions

LOG = logging.getLogger(__name__)


def create_vae_pz_model(encoder, decoder, input_shape, latent_dim):
    encoder.trainable = False
    # Link the models
    x_input = Input(shape=(input_shape))
    z = tfp.layers.MultivariateNormalTriL(
        latent_dim,
        name="latent_space",
        convert_to_tensor_fn=tfd.Distribution.sample,
    )(encoder(x_input))

    pz_estimator = create_pz_estimator(latent_dim=latent_dim)

    return Model(inputs=x_input, outputs=(decoder(z), pz_estimator(z))), pz_estimator


def train_pz(
    input_shape,
    encoder,
    decoder,
    train_generator,
    validation_generator,
    callbacks,
    loss_function,
    latent_dim=16,
    optimizer=tf.keras.optimizers.Adam(1e-4),
    epochs=35,
    verbose=1,
):
    """Train only the the components of VAE model (encoder and/or decoder).

    Parameters
    ----------
    train_generator:
        generator to be used for training the network.
        keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
    validation_generator:
        generator to be used for validation
        keras.utils.Sequence returning (inputs, targets) or (inputs, targets, sample_weights)
    callbacks: list
        List of keras.callbacks.Callback instances.
        List of callbacks to apply during training.
        See tf.keras.callbacks
    loss_function: python function
        function that can compute the loss.
    optimizer: str or tf.keras.optimizers
        String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
    epochs: int
        number of epochs for which the model is going to be trained
    verbose: int
        verbose option for training.
        'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
        Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively (eg, in a production environment).

    """
    # encoder.summary()

    # LOG.info("Initial learning rate: " + str(lr))

    LOG.info("Number of epochs: " + str(epochs))

    encoder.trainable = True
    decoder.trainable = True
    linked_pz_model, pz_estimator = create_vae_pz_model(
        encoder, decoder, input_shape, latent_dim
    )

    if loss_function is None:
        print("pass valid loss function")
    linked_pz_model.compile(
        optimizer=optimizer,
        loss=loss_function,
        experimental_run_tf_function=False,
    )
    print(linked_pz_model.summary())
    print(pz_estimator.summary())
    hist = linked_pz_model.fit(
        x=train_generator[0] if isinstance(train_generator, tuple) else train_generator,
        y=train_generator[1] if isinstance(train_generator, tuple) else None,
        epochs=epochs,
        verbose=verbose,
        shuffle=True,
        validation_data=validation_generator,
        callbacks=callbacks,
        workers=8,
        use_multiprocessing=True,
    )
    return hist
