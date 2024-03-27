"""Perform SSI."""


def ssi(btk_draw_generator, isolated_galaxy):
    """Inject sources into the stamp.
    """
    blend = next(btk_draw_generator)
    source_injected = blend.isolated_images[:, 0] + isolated_galaxy
    return source_injected, blend
