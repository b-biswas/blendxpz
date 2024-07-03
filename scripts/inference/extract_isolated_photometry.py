import logging
import os
import sys

import numpy as np
import pandas as pd
import sep
import yaml

from blendxpz.metrics import compute_aperture_photometry
from blendxpz.simulations.btk_setup import btk_setup_helper
from blendxpz.utils import get_blendxpz_config_path, get_madness_config_path

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)

# Take inputs

LOG.info(sys.argv)
dataset = sys.argv[1]  # should be either training or validation
if dataset not in ["training", "validation"]:
    raise ValueError(
        "The first argument (dataset) should be either training or validation"
    )

blend_type = sys.argv[2]  # set to 4 to generate blended scenes
if blend_type not in ["isolated", "blended"]:
    raise ValueError("The second argument should be either isolated or blended")


with open(get_blendxpz_config_path()) as f:
    blendxpz_config = yaml.safe_load(f)

with open(get_madness_config_path()) as f:
    madness_config = yaml.safe_load(f)

survey_name = blendxpz_config["SURVEY_NAME"]
btksims_config = madness_config["btksims"]

# set the save path
SAVE_PATH = os.path.join(
    btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name], blend_type + "_" + dataset
)

# definte the survey
_, _, survey = btk_setup_helper(
    survey_name=survey_name,
    btksims_config=btksims_config,
)


def PopulateFileList(data_folder):
    """Populate file list.

    Parameters
    ----------
    data_folder: string
        Path to the folder with .npy files

    Returns
    -------
    list_of_images: list
        names of all images in the data_folder.

    """
    list_of_images = []
    for root, dirs, files in os.walk(data_folder, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.splitext(file_path)[1] != ".npy":
                continue
            list_of_images.append(file_path)
    # Make sure that some data exists
    assert len(list_of_images) > 0

    LOG.info(f"File list is populated .. there are {len(list_of_images)} files")

    return list_of_images


def compute_photometry_data(data_folder):
    """Yield examples.

    Parameters
    ----------
    data_folder: string
        path to the data folder

    """
    list_of_images = PopulateFileList(data_folder)
    galaxies = []

    for gal_file in list_of_images:
        # very verbose
        LOG.info(f"File : {gal_file} is being treated")
        current_sample = np.load(gal_file, allow_pickle=True)
        key = os.path.splitext(gal_file)[0]

        bkg_rms = {}
        for band in range(len(survey.available_filters)):  # type: ignore
            bkg_rms[band] = sep.Background(
                current_sample["blended_gal_stamps"][0][band]
            ).globalrms

        galaxy_info = compute_aperture_photometry(
            field_image=current_sample["blended_gal_stamps"][0].astype("float32"),
            predictions=None,
            xpos=[22],
            ypos=[22],
            a=[current_sample["flux_radius"][0]],
            b=[current_sample["flux_radius"][0]],
            theta=[0],
            bkg_rms=bkg_rms,
            survey=survey,
        )

        galaxy_info["flux_radius"] = current_sample["flux_radius"][0]
        galaxy_info["pz"] = current_sample["pz"][0].astype("float32")
        galaxy_info["key"] = key
        galaxy_info = pd.DataFrame(galaxy_info)

        galaxies.append(galaxy_info)

    galaxies = pd.concat(galaxies)

    return galaxies


galaxies = compute_photometry_data(SAVE_PATH)
galaxies.to_pickle(os.path.join(SAVE_PATH, "photometry_data.pkl"))
