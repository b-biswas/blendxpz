import logging
import os
import pickle
import sys

import astropy.units as u
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from madness_deblender.callbacks import define_callbacks

from blendxpz.pz_estimators.mlp import create_mpl_estimator
from blendxpz.simulations.btk_setup import btk_setup_helper
from blendxpz.utils import (
    column_order,
    get_blendxpz_config_path,
    get_data_dir_path,
    get_madness_config_path,
)

tfd = tfp.distributions


# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)


# loss func
def pz_loss_function(y, predicted):
    # return -tfp.distributions.Normal(predicted[0], predicted[1]).log_prob(y)
    return tf.math.abs(y - predicted) / (y + 1)


# Take inputs
blend_type = sys.argv[1]  # should be either isolated or blended
if blend_type not in ["isolated", "blended"]:
    raise ValueError("The second argument should be either isolated or blended")

with open(get_blendxpz_config_path()) as f:
    blendxpz_config = yaml.safe_load(f)

with open(get_madness_config_path()) as f:
    madness_config = yaml.safe_load(f)

survey_name = blendxpz_config["SURVEY_NAME"]
if survey_name not in ["LSST", "HSC"]:
    raise ValueError("survey should be one of: LSST or HSC")

# define the parameters
batch_size = 100
epochs = 200
lr_scheduler_epochs = 50
linear_norm_coeff = 10000
patience = 25

# load survey
_, _, survey = btk_setup_helper(
    survey_name=survey_name,
)

# get training data
train_data = {}
val_data = {}
BASE_DATA_PATH = madness_config["btksims"]["TRAIN_DATA_SAVE_PATH"][survey_name]

dataset = "training"
data_path = data_path = os.path.join(
    BASE_DATA_PATH, blend_type + "_" + dataset, "photometry_data.pkl"
)
with open(data_path, "rb") as pickle_file:
    file_data = pickle.load(pickle_file)

for band in survey.available_filters:

    z_point = survey.get_filter(band).zeropoint
    exp_time = survey.get_filter(band).full_exposure_time

    actual_phot_flux = file_data[f"{band}_phot_flux"].values
    file_data[f"{band}_phot_mag"] = (
        (actual_phot_flux * u.electron / exp_time).to(u.mag(u.electron / u.s)) + z_point
    ).value

file_data = file_data.dropna()

# First compute the mean and std of training set for normalization.

norms = {}
norms["mu"] = {}
norms["sigma"] = {}
for filter in survey.available_filters:

    z_point = survey.get_filter(filter).zeropoint
    exp_time = survey.get_filter(filter).full_exposure_time

    # mask = file_data[f"{filter}_phot_flux"] / file_data[f"{filter}_phot_fluxerrs"] > 10
    norms["mu"][f"{filter}"] = np.mean(file_data[f"{filter}_phot_mag"].values)
    norms["sigma"][f"{filter}"] = np.std(file_data[f"{filter}_phot_mag"].values)


norms_path = os.path.join(BASE_DATA_PATH, blend_type + "_" + dataset, "norms.pkl")
with open(norms_path, "wb") as f:
    pickle.dump(norms, f)

# create and normalize the training and validation data

for dataset in ["training", "validation"]:
    data_path = os.path.join(
        BASE_DATA_PATH, blend_type + "_" + dataset, "photometry_data.pkl"
    )
    with open(data_path, "rb") as pickle_file:
        file_data = pickle.load(pickle_file)

    for band in survey.available_filters:

        z_point = survey.get_filter(band).zeropoint
        exp_time = survey.get_filter(band).full_exposure_time

        actual_phot_flux = file_data[f"{band}_phot_flux"].values
        file_data[f"{band}_phot_mag"] = (
            (actual_phot_flux * u.electron / exp_time).to(u.mag(u.electron / u.s))
            + z_point
        ).value

    file_data = file_data.dropna()

    data = {}
    data["x"] = {}
    mask = file_data[f"{filter}_phot_flux"] / file_data[f"{filter}_phot_fluxerrs"] > 10
    for filter in survey.available_filters:

        z_point = survey.get_filter(filter).zeropoint
        exp_time = survey.get_filter(filter).full_exposure_time

        actual_phot_mag = file_data[f"{filter}_phot_mag"].values
        data["x"][f"{filter}_phot_mag"] = (
            file_data[f"{filter}_phot_mag"].values - norms["mu"][f"{filter}"]
        ) / norms["sigma"][f"{filter}"]

    data["x"]["flux_radius"] = file_data["flux_radius"].values / 10
    data["x"] = pd.DataFrame(data["x"])

    data["y"] = file_data["pz"]

    if dataset == "training":
        train_data = data
    else:
        val_data = data


# create model
mlp_estimator = create_mpl_estimator(num_filters=len(survey.available_filters) + 1)

# Keras Callbacks
data_path = get_data_dir_path()

# path_weights = os.path.join(data_path, f"catsim_kl{kl_weight_exp}{latent_dim}d")
path_weights = os.path.join(data_path, "models", survey_name + "_mlp_estimator")
callbacks = define_callbacks(
    os.path.join(path_weights),
    lr_scheduler_epochs=lr_scheduler_epochs,
    patience=patience,
)

mlp_estimator.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4, clipvalue=1),
    loss=pz_loss_function,
    experimental_run_tf_function=False,
)

hist = mlp_estimator.fit(
    x=train_data["x"][column_order(survey)].to_numpy(),
    y=train_data["y"].to_numpy(),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    shuffle=True,
    validation_data=(val_data["x"].to_numpy(), val_data["y"].to_numpy()),
    callbacks=callbacks,
    workers=8,
    use_multiprocessing=True,
)

np.save(path_weights + "/train_vae_history.npy", hist.history)
