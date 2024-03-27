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
        
        x_peak, y_peak = 0
        possible_values = np.array([-self.shift, 0, self.shift])
        while (x_peak==0) & (y_peak ==0):
            x_peak = np.choice(possible_values)
            y_peak = np.choice(possible_values)

        blend_table["ra"] += x_peak
        blend_table["dec"] += y_peak

        if np.any(blend_table["ra"] > self.stamp_size / 2.0) or np.any(
            blend_table["dec"] > self.stamp_size / 2.0
        ):
            warnings.warn("Object center lies outside the stamp")
        return blend_table