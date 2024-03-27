"""Perform SSI."""

from blendxpz.simulations.sampling import FixedDistSampling

def ssi_helper(btk_draw_generator, isolated_galaxy):
    blend=next(btk_draw_generator)
    source_injected = blend.isolated_images[:, 0] + isolated_galaxy
    return source_injected