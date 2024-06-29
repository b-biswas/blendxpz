"""Simulate test dataset."""

import logging
import os
import pickle
import sys

import yaml

from blendxpz.simulations.btk_setup import btk_setup_helper
from blendxpz.simulations.sampling import CustomSampling
from blendxpz.utils import get_blendxpz_config_path, get_madness_config_path

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)

LOG = logging.getLogger(__name__)

density = sys.argv[1]

if density not in ["high", "low"]:
    raise ValueError("The first argument should be either high or low")

with open(get_blendxpz_config_path()) as f:
    blendxpz_config = yaml.safe_load(f)

with open(get_madness_config_path()) as f:
    madness_config = yaml.safe_load(f)

survey_name = blendxpz_config["SURVEY_NAME"]
LOG.info(f"survey: {survey_name}")
btksims_config = madness_config["btksims"]
simulation_path = btksims_config["TEST_DATA_SAVE_PATH"][survey_name]

catalog, generator, survey = btk_setup_helper(
    survey_name=survey_name,
    btksims_config=btksims_config,
)

sim_config = btksims_config["TEST_PARAMS"]
LOG.info(f"Simulation config: {sim_config}")

index_range = [sim_config[survey_name]["index_start"], len(catalog.table)]
sampling_function = CustomSampling(
    index_range=index_range,
    min_number=sim_config[density + "_density"]["min_number"],
    max_number=sim_config[density + "_density"]["max_number"],
    maxshift=sim_config["maxshift"],
    stamp_size=sim_config["stamp_size"],
    seed=sim_config["btk_seed"],
    unique=sim_config["unique_galaxies"],
    dataset="test",
    pixel_scale=survey.pixel_scale.value,
)

draw_generator = generator(
    catalog,
    sampling_function,
    survey,
    batch_size=sim_config["btk_batch_size"],
    stamp_size=sim_config["stamp_size"],
    njobs=16,
    add_noise="all",
    verbose=False,
    seed=sim_config["btk_seed"],
)

for file_num in range(sim_config[survey_name]["num_files"]):
    print("Processing file " + str(file_num))
    blend = next(draw_generator)

    save_file_name = os.path.join(
        simulation_path,
        density,
        str(file_num) + ".pkl",
    )
    print(save_file_name)
    with open(save_file_name, "wb") as pickle_file:
        pickle.dump(blend, pickle_file)
