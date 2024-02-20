import tensorflow as tf 

@tf.function
def pz_loss(pz, predicted_pz):
    #loss = -tfd.Normal(loc=predicted_pz[:, 0], scale=predicted_pz[:, 1]).log_prob(pz)
    loss = (predicted_pz-pz)**2
    return tf.reduce_mean(loss)