import tensorflow as tf


@tf.function
def pz_loss(pz, predicted_pz):
    # loss = -tfd.Normal(loc=predicted_pz[:, 0], scale=predicted_pz[:, 1]).log_prob(pz)
    loss = tf.abs(predicted_pz - pz)
    return tf.reduce_mean(loss)


@tf.function
def deblender_loss_fn_wrapper(
    sigma_cutoff, use_pz_reg=False, ch_alpha=None, linear_norm_coeff=10000
):
    """Input field sigma into ssim loss function.

    Parameters
    ----------
    sigma_cutoff: list
        list of sigma levels (normalized) in the bands.
    use_ssim: bool
        Flag to add an extra ssim term to the loss function.
        This loss is supposed to force the network to learn information in noisy bands relatively earlier.
    ch_alpha: madness.callbacks.ChangeAlpha
        instance of ChangeAlpha to update the weight of SSIM over epochs.
    linear_norm_coeff: int
        linear norm coefficient used for normalizing.

    Returns
    -------
    deblender_ssim_loss_fn:
        function to compute the loss using SSIM weight.

    """

    @tf.function
    def deblender_ssim_loss_fn(y, predicted):
        """Compute the loss under predicted distribution, weighted by the SSIM.

        Parameters
        ----------
        y: array/tensor
            Galaxy ground truth.
        predicted_galaxy: tf tensor
            pixel wise prediction of the flux.

        Returns
        -------
        objective: float
            objective to be minimized by the minimizer.

        """
        galaxy, pz = y
        predicted_galaxy, predicted_pz = predicted
        loss = tf.reduce_sum(
            (galaxy - predicted_galaxy) ** 2
            / (sigma_cutoff**2 + galaxy / linear_norm_coeff),
            axis=[1, 2, 3],
        )

        if use_pz_reg:
            pz_loss = tf.abs(pz - predicted_pz)
            tf.stop_gradient(ch_alpha.alpha)
            loss = loss + ch_alpha.alpha * pz_loss

        loss = tf.reduce_mean(loss)
        # if ch_alpha.alpha > 0:
        # band_normalizer = tf.reduce_max(y+1e-9, axis=[1, 2], keepdims=True)
        # beta_factor = 2.5
        # tf.stop_gradient(ch_alpha.alpha)
        # loss2 = tf.keras.backend.binary_crossentropy(
        #     tf.math.tanh(
        #         tf.math.asinh(
        #             beta_factor * (predicted_galaxy / band_normalizer)
        #         )
        #     ),
        #     tf.math.tanh(
        #         tf.math.asinh(
        #             beta_factor * (y / band_normalizer)
        #         )
        #     ),
        # )  # computes the mean across axis 0
        # loss2 = tf.reduce_sum(loss2)
        # loss = loss2
        # weight = tf.math.reduce_max(x, axis= [1, 2])
        # objective = tf.math.reduce_sum(loss, axis=[1, 2])
        # weighted_objective = -tf.math.reduce_mean(tf.divide(objective, weight))

        return loss

    return deblender_ssim_loss_fn
