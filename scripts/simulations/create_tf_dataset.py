"""First call to generate tf datasets."""

import os

import yaml
from blendxpz.training.dataset_generator import loadExCOSMOSDataset

from blendxpz.utils import get_blendxpz_config_path, get_madness_config_path


with open(get_blendxpz_config_path()) as f:
    blendxpz_config = yaml.safe_load(f)

with open(get_madness_config_path()) as f:
    madness_config = yaml.safe_load(f)

survey_name = blendxpz_config["SURVEY_NAME"]
btksims_config = madness_config["btksims"]

loadExCOSMOSDataset(
    train_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "blended_training",
    ),
    val_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "blended_validation",
    ),
    output_dir=os.path.join(
        blendxpz_config["TF_DATASET_PATH"][survey_name], "blended_tfDataset"
    ),
)

# loadExCOSMOSDataset(
#     train_data_dir=os.path.join(
#         btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
#         "isolated_training",
#     ),
#     val_data_dir=os.path.join(
#         btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
#         "isolated_validation",
#     ),
#     output_dir=os.path.join(
#         blendxpz_config["TF_DATASET_PATH"][survey_name],
#         "isolated_tfDataset",
#     ),
# )