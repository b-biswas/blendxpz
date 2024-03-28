"""SSI in a stamp"""


def ssi_on_simulations(btk_draw_generator, isolated_galaxy):
    """Inject sources into a simulated stamp.

    Parameters
    """
    blend = next(btk_draw_generator)
    source_injected = blend.isolated_images[:, 0] + isolated_galaxy
    return source_injected, blend


def ssi_on_realdata():
    pass
