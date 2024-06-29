import astropy.units as u
import btk
from astropy.table import Table


def HSC_filter_snr_adjust(survey):
    filters = survey.available_filters
    offsets = [-0.15, 0.15, 0.75, 1.4, 0.5]
    for i, f in enumerate(filters):
        filt = survey.get_filter(f)
        filt.sky_brightness = (
            (filt.sky_brightness.value + offsets[i]) * u.mag / (u.arcsec) ** 2
        )

    return survey


def btk_setup_helper(survey_name, btksims_config):
    survey = btk.survey.get_surveys(survey_name)
    if survey_name == "HSC":
        survey = HSC_filter_snr_adjust(survey)

    CATALOG_PATH = btksims_config["CAT_PATH"][survey_name]

    if type(CATALOG_PATH) == list:
        catalog = btk.catalog.CosmosCatalog.from_file(
            CATALOG_PATH, exclusion_level="none"
        )
        generator = btk.draw_blends.CosmosGenerator
    else:
        catalog = btk.catalog.CatsimCatalog.from_file(CATALOG_PATH)
        generator = btk.draw_blends.CatsimGenerator

    catalog.table = Table.from_pandas(
        catalog.table.to_pandas().sample(frac=1, random_state=0).reset_index(drop=True)
    )

    return catalog, generator, survey
