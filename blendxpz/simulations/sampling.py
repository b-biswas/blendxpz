"""Define custom sampling for BTK"""

import logging
import warnings

import numpy as np
from btk.sampling_functions import SamplingFunction, _get_random_center_shift
from btk.utils import DEFAULT_SEED

# logging level set to INFO
LOG = logging.getLogger(__name__)


class FixedDistSampling(SamplingFunction):
    """Default sampling function used for producing blend tables."""

    def __init__(
        self,
        index_range,
        shift,
        max_number=2,
        min_number=1,
        stamp_size=24.0,
        unique=True,
        seed=DEFAULT_SEED,
    ):
        """Initialize default sampling function.

        Parameters
        ----------
        max_number: int
            maximum number of galaxies in the stamp.
        min_number: int
            minimum number of galaxies in the stamp.
        stamp_size: (=float
            Size of the desired stamp.
        index_range: tuple
            range to indexes to sample galaxies from the catalog.
        shift: int
            Magnitude of the the shift.
            eg. if shift =1, along each coordinate, distace can be either one or 1,
            but not both zeroes at the same time.
        unique: bool
            whether to have unique galaxies in different stamps.
            If true, galaxies are sampled sequentially from the catalog.
        seed: int
            Seed to initialize randomness for reproducibility.

        """
        super().__init__(max_number=max_number, min_number=min_number, seed=seed)
        self.stamp_size = stamp_size
        self.shift = shift
        self.index_range = index_range
        self.unique = unique
        self.indexes = list(np.arange(index_range[0], index_range[1]))

    @property
    def compatible_catalogs(self):
        """Tuple of compatible catalogs for this sampling function."""
        return "CatsimCatalog", "CosmosCatalog"

    def __call__(self, table, indexes=None):
        """Apply default sampling to the input CatSim-like catalog.

        Returns an astropy table with entries corresponding to a blend centered close to postage
        stamp center.

        Function selects entries from the input table such that the number of objects per blend
        is set at a random integer ``self.min_number`` and ``self.max_number`` (both inclusive).
        If ``unique`` is set to True, the blend table the table is sampled sequentially, untill
        all galaxies a exhausted. While the table entries are sampled randomly ``unique`` is False.
        The centers are randomly places such that the distance from the center is one pixel
        (in each direction including diagonally).

        Parameters
        ----------
        table: astropy.Table
            Table containing entries corresponding to galaxies from which to sample.
        indexes: list
            Contains the indexes of the galaxies to use.

        Returns
        -------
            Astropy.table with entries corresponding to one blend.

        """
        number_of_objects = self.rng.integers(self.min_number, self.max_number + 1)

        if indexes is None:
            if self.unique:
                if number_of_objects > len(self.indexes):
                    raise ValueError(
                        "Too many iterations. All galaxies have been sampled."
                    )
                current_indexes = self.indexes[:number_of_objects]
                self.indexes = self.indexes[number_of_objects:]

            else:
                current_indexes = self.rng.choice(self.indexes, size=number_of_objects)
            # print(current_indexes)
            blend_table = table[current_indexes]
        else:
            blend_table = table[indexes]
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0

        # The galaxy will be on the boundary of the circle of the given radius
        x_peak = self.rng.uniform(-self.shift, self.shift)
        y_peak = self.rng.choice([1, -1]) * (self.shift**2 - x_peak**2) ** 0.5

        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table


def check_repeated_pixel(x_peak, y_peak, pixel_scale, maxshift):
    """Check repeated pixel.

    Parameters
    ----------
    x_peak : float
        x_peak in arc seconds
    y_peak : float
        y_peak in arc seconds
    pixel_scale : float/int
        pixel_scale of the survey
    maxshift : float
        maxshift in arc seconds

    """
    dim = int(2 * maxshift / pixel_scale + 1)
    centers = np.zeros((dim, dim))
    for x, y in zip(x_peak, y_peak):
        x = int(np.round(x / pixel_scale) + maxshift / pixel_scale)
        y = int(np.round(y / pixel_scale) + maxshift / pixel_scale)
        centers[x][y] += 1
    return True if np.sum(centers > 1) != 0 else False

class CustomSampling(SamplingFunction):
    """Default sampling function used for producing blend tables."""

    def __init__(
        self,
        index_range,
        max_number=2,
        min_number=1,
        stamp_size=24.0,
        maxshift=None,
        unique=True,
        seed=DEFAULT_SEED,
        dataset="train_val",
        pixel_scale=0,
    ):
        """Initialize default sampling function.

        Parameters
        ----------
        max_number: int
            maximum number of galaxies in the stamp.
        min_number: int
            minimum number of galaxies in the stamp.
        stamp_size: float
            Size of the desired stamp.
        index_range: tuple
            range to indexes to sample galaxies from the catalog.
        maxshift: float
            Magnitude of the maximum value of shift. If None then it
            is set as one-tenth the stamp size. (in arcseconds)
        unique: bool
            whether to have unique galaxies in different stamps.
            If true, galaxies are sampled sequentially from the catalog.
        seed: int
            Seed to initialize randomness for reproducibility.
        dataset: str
            either 'train_val' or 'test'
        pixel_scale: int
            survey pixel scale requred for test dataset to make sure centers don't lie in same pixel.

        """
        super().__init__(max_number=max_number, min_number=min_number, seed=seed)
        self.stamp_size = stamp_size
        self.maxshift = maxshift if maxshift else self.stamp_size / 10.0
        self.index_range = index_range
        self.unique = unique
        self.indexes = list(np.arange(index_range[0], index_range[1]))

        if dataset not in ["train_val", "test"]:
            raise ValueError("dataset can only be either `train_val` or `test`")

        if dataset == "test":
            if pixel_scale == 0:
                raise ValueError("Pass appropriate pixel scale")
        self.pixel_scale = pixel_scale
        self.dataset = dataset

    @property
    def compatible_catalogs(self):
        """Tuple of compatible catalogs for this sampling function."""
        return "CatsimCatalog", "CosmosCatalog"

    def __call__(self, table, shifts=None, indexes=None):
        """Apply default sampling to the input CatSim-like catalog.

        Returns an astropy table with entries corresponding to a blend centered close to postage
        stamp center.

        Function selects entries from the input table that are brighter than 25.3 mag
        in the i band. Number of objects per blend is set at a random integer
        between 1 and ``self.max_number``. The blend table is then randomly sampled
        entries from the table after selection cuts. The centers are randomly
        distributed within 1/10th of the stamp size. Here even though the galaxies
        are sampled from a CatSim catalog, their spatial location are not
        representative of real blends.

        Parameters
        ----------
        table: astropy.Table
            Table containing entries corresponding to galaxies from which to sample.
        shifts: list
            Contains arbitrary shifts to be applied instead of random ones.
            Should of the form [x_peak,y_peak] where x_peak and y_peak are the lists
            containing the x and y shifts.
        indexes: list
            Contains the indexes of the galaxies to use.

        Returns
        -------
            Astropy.table with entries corresponding to one blend.

        """
        number_of_objects = self.rng.integers(self.min_number, self.max_number + 1)

        if indexes is None:
            if self.unique:
                if number_of_objects > len(self.indexes):
                    raise ValueError(
                        "Too many iterations. All galaxies have been sampled."
                    )
                current_indexes = self.indexes[:number_of_objects]
                self.indexes = self.indexes[number_of_objects:]

            else:
                current_indexes = self.rng.choice(self.indexes, size=number_of_objects)
            # print(current_indexes)
            blend_table = table[current_indexes]
        else:
            blend_table = table[indexes]
        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0
        if shifts is None:
            x_peak, y_peak = _get_random_center_shift(
                number_of_objects, self.maxshift, self.rng
            )
            if self.dataset == "test":
                while check_repeated_pixel(
                    x_peak, y_peak, pixel_scale=self.pixel_scale, maxshift=self.maxshift
                ):
                    LOG.info("Repeated centres, sampling again...")
                    # print(pd.DataFrame({'x': x_peak, 'y': y_peak})) # just to see if they were repeated
                    x_peak, y_peak = _get_random_center_shift(
                        number_of_objects, self.maxshift, self.rng
                    )
        else:
            x_peak, y_peak = shifts
        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table
