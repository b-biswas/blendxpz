import os


def get_data_dir_path():
    """Fetch path to the data folder of maddeb.

    Returns
    -------
    data_dir: str
        path to data folder

    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curdir, "data")

    return data_dir


def get_blendxpz_config_path():
    """Fetch path to madness_deblender config yaml file.

    Returns
    -------
    data_dir: str
        path to data folder

    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    blendxpz_config_path = os.path.join(curdir, "configs", "blendxpz_config.yaml")

    return blendxpz_config_path


def get_madness_config_path():
    """Fetch path to madness_deblender config yaml file.

    Returns
    -------
    data_dir: str
        path to data folder

    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    madness_config_path = os.path.join(curdir, "configs", "madness_config.yaml")

    return madness_config_path


def column_order(survey):
    col_names = []
    for filter in survey.available_filters:
        col_names.append(f"{filter}_phot_mag")
    col_names.append("flux_radius")
    return col_names
