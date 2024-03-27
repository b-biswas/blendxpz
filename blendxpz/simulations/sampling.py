"""Define custom sampling for BTK"""

import warnings

import numpy as np
from btk.sampling_functions import SamplingFunction
from btk.utils import DEFAULT_SEED


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

        x_peak = self.rng.uniform(-self.shift, self.shift)
        y_peak = (self.shift**2 - x_peak**2) ** 0.5

        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table
