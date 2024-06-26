"""Train pasquet et al. model."""

import logging
import os
import sys

import galcheat
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from madness_deblender.callbacks import define_callbacks

from blendxpz.utils import get_data_dir_path
from blendxpz.pz_estimators.CNN_pz_estimator import create_model
from blendxpz.training.dataset_generator import batched_ExCOSMOS
from blendxpz.utils import get_madness_config_path

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

# define the parameters
batch_size = 100
epochs = 200
lr_scheduler_epochs = 30

linear_norm_coeff = 1
patience = 30

def pz_loss_function(y, predicted):
    #return -tfp.distributions.Normal(predicted[0], predicted[1]).log_prob(y)
    return (y-predicted)**2

with open(get_madness_config_path()) as f:
    madness_config = yaml.safe_load(f)

survey_name = madness_config["SURVEY_NAME"]

if survey_name not in ["LSST", "HSC"]:
    raise ValueError(
        "survey should be one of: LSST or HSC"
    )  # other surveys to be added soon!


survey = galcheat.get_survey(survey_name)

# Keras Callbacks
data_path = get_data_dir_path()

# path_weights = os.path.join(data_path, f"catsim_kl{kl_weight_exp}{latent_dim}d")
path_weights = os.path.join(data_path, survey.name + "_CNN_estimator")
callbacks = define_callbacks(
    os.path.join(path_weights),
    lr_scheduler_epochs=lr_scheduler_epochs,
    patience=patience,
)

pz_model = create_model()
# Define the generators
ds_isolated_train, ds_isolated_val = batched_ExCOSMOS(
    train_data_dir=None,
    val_data_dir=None,
    tf_dataset_dir=os.path.join(
        madness_config["TF_DATASET_PATH"][survey_name], "isolated_tfDataset"
    ),
    linear_norm_coeff=1,
    batch_size=batch_size,
    x_col_name="isolated_gal_stamps",
    y_col_name="pz",
)

pz_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-8, clipvalue=0.1),
            loss=pz_loss_function,
            experimental_run_tf_function=False,
        )

hist = pz_model.fit(
            x=ds_isolated_train,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=ds_isolated_val,
            callbacks=callbacks,
            workers=8,
            use_multiprocessing=True,
)

np.save(path_weights + "/train_vae_history.npy", hist.history)