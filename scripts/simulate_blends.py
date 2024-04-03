import glob
import os
import random

import astropy.units as u
import btk
import yaml
from astropy.io import fits
from astropy.table import vstack

from blendxpz.simulations.sampling import FixedDistSampling
from blendxpz.simulations.ssi import ssi
from blendxpz.utils import get_blendxpz_config_path, get_data_dir_path

with open(get_blendxpz_config_path()) as f:
    blendxpz_config = yaml.safe_load(f)
data_dir_path = get_data_dir_path()

survey_name = blendxpz_config["SURVEY_NAME"]
config = blendxpz_config[survey_name]
stamp_pixel_size = config["STAMP_PIXEL_SIZE"]

random.seed = config["DEFAULT_SEED"]

galaxy_image_paths = glob.glob(
    os.path.join(data_dir_path, "**", "images", "*.fits"), recursive=True
)
if config["NUM_REAL_GAL_TO_USE"] > len(galaxy_image_paths):
    raise ValueError("Not enough real galaxies.")
random.shuffle(galaxy_image_paths)

# Set up BTK parameters
pixel_shift_distance = 10

CATALOG_DIR = os.path.join(data_dir_path, config["GALSIM_COSMOS_DATA_DIR"])
CATALOG_NAMES = config["CATALOG_NAMES"]
CATALOG_PATHS = [
    os.path.join(CATALOG_DIR, CATALOG_NAME) for CATALOG_NAME in CATALOG_NAMES
]
survey = btk.survey.get_surveys(survey_name)
btk_stamp_size = survey.pixel_scale.value * stamp_pixel_size
min_number = 1
max_number = 1
batch_size = config["NUM_BLENDS_PER_GAL"]
shift = pixel_shift_distance * survey.pixel_scale.value

survey = btk.survey.get_surveys("HSC")

# setup HSC survey for real data
filters = survey.available_filters

catalog = btk.catalog.CosmosCatalog.from_file(CATALOG_PATHS, exclusion_level="none")
generator = btk.draw_blends.CosmosGenerator

seed = 13

sampling_function = FixedDistSampling(
    index_range=[0, 20],
    shift=shift,
    min_number=min_number,
    max_number=max_number,
    stamp_size=btk_stamp_size,
    seed=seed,
    unique=False,
)

# Specify all parameters except the PSF (depends upon the coadd we sample from)
for f in filters:
    filt = survey.get_filter(f)
    filt.zeropoint = 27 * u.mag
    filt.full_exposure_time = 1 * u.s


# Start simulating galaxies
for real_gal_num, real_galaxy_num in enumerate(config["NUM_REAL_GAL_TO_USE"]):

    gal_file_path = galaxy_image_paths[real_gal_num]
    gal_file_name = gal_file_path.split("/")[-1]
    # Sample a coadd
    tract, patch, gal_number = gal_file_name.split("_")

    isolated_galaxy = fits.getdata(gal_file_path)

    # extract a given size
    size = isolated_galaxy.shape[1]
    isolated_galaxy = isolated_galaxy[
        :,
        size // 2 - stamp_pixel_size // 2 : size // 2 + stamp_pixel_size // 2 + 1,
        size // 2 - stamp_pixel_size // 2 : size // 2 + stamp_pixel_size // 2 + 1,
    ]

    # Set the PSF
    psf_dir = os.path.join(
        data_dir_path, config["REAL_DATA_DIR"], tract + "_" + patch, "psfs"
    )
    for f in filters:
        filt.psf = lambda: btk.survey.get_psf_from_file(
            os.path.join(psf_dir, f), survey
        )

    # Set up BTK blend generator
    draw_generator = generator(
        catalog,
        sampling_function,
        survey,
        batch_size=batch_size,
        stamp_size=btk_stamp_size,
        njobs=1,
        add_noise="all",
        verbose=False,
        seed=seed,
    )

    # run SSI
    ssi_galaxies, blend = ssi(draw_generator, isolated_galaxy)

    # Now save files
    data = vstack(blend.catalog_list)
    data["blend"] = ssi_galaxies

    gal_key = gal_file_name.split(".")[0]

    # Save blended scenes
    save_file_name = os.path.join(
        data_dir_path,
        config["SIMULATION_SAVE_DIR"],
        "blended_" + gal_key + ".pkl",
    )
    data.write(save_file_name, format="fits", overwrite=True)

    print(f"Saving Galaxy: {gal_key}")
