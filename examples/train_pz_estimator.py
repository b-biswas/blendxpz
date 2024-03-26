"""Train all models."""

import logging
import os
import sys

import galcheat
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from galcheat.utilities import mean_sky_level

from madness_deblender.callbacks import define_callbacks
from madness_deblender.dataset_generator import batched_CATSIMDataset_pz
from madness_deblender.FlowVAEnet import FlowVAEnet

import madness_deblender.utils as madness_deblender_utils
#from madness_deblender.utils import get_data_dir_path, get_madness_deblender_config_path

from bpz.losses import pz_loss
from bpz.pz_estimator import train_pz
import bpz.utils as bpz_utils



tfd = tfp.distributions

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

# define the parameters
batch_size = 100
vae_epochs = 200
flow_epochs = 200
deblender_epochs = 150
lr_scheduler_epochs = 30
latent_dim = 16
linear_norm_coeff = 10000
patience = 30

with open(madness_deblender_utils.get_madness_deblender_config_path()) as f:
    madness_deblender_config = yaml.safe_load(f)

survey_name = madness_deblender_config["survey_name"]

if survey_name not in ["LSST", "HSC"]:
    raise ValueError(
        "survey should be one of: LSST or HSC"
    )  # other surveys to be added soon!

train_models = sys.argv[
    1
]  # either "all" or a list contraining: ["GenerativeModel","NormalizingFlow","Deblender"]
kl_weight_exp = int(sys.argv[1])
kl_weight = 10**-kl_weight_exp
LOG.info(f"KL weight{kl_weight}")

survey = galcheat.get_survey(survey_name)

kl_prior = tfd.Independent(
    tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1
)
# Keras Callbacks
loadweights_path = madness_deblender_utils.get_data_dir_path()
saveweights_path = bpz_utils.get_data_dir_path()

# path_weights = os.path.join(data_path, f"catsim_kl{kl_weight_exp}{latent_dim}d")
path_loadweights = os.path.join(loadweights_path, survey.name + str(kl_weight))
path_saveweights = os.path.join(saveweights_path, survey.name + str(kl_weight))
print(path_loadweights)

# Define the generators
ds_isolated_train, ds_isolated_val = batched_CATSIMDataset_pz(
    train_data_dir=None,
    val_data_dir=None,
    tf_dataset_dir=os.path.join(
        madness_deblender_config["TF_DATASET_PATH"][survey_name], "isolated_tfDataset"
    ),
    linear_norm_coeff=linear_norm_coeff,
    batch_size=batch_size,
    x_col_name="blended_gal_stamps",
)

callbacks = define_callbacks(
    os.path.join(path_saveweights, "pz"),
    lr_scheduler_epochs=lr_scheduler_epochs,
    patience=vae_epochs,
)

f_net = FlowVAEnet(survey=survey)
f_net.load_vae_weights(
    weights_path=os.path.join(path_loadweights, "vae/val_loss")
)

hist_pz = train_pz(
    input_shape=[45, 45, 5],
    encoder=f_net.encoder,
    train_generator=ds_isolated_train,
    validation_generator=ds_isolated_val,
    callbacks=callbacks,
    loss_function=pz_loss,
    latent_dim = 16,    
    optimizer=tf.keras.optimizers.Adam(1e-5, clipvalue=0.1),
    epochs=100,
    verbose=1,
)

np.save(path_saveweights + "/train_pz_history.npy", hist_pz.history)
