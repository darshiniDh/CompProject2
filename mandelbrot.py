import numpy as np

def get_escape_time(c: complex, max_iterations: int) -> int | None:
    """
    :param c: Complex number
    :param max_iterations: Specified number of iterations

    :return: The number of iterations which pass before c escapes.
    If c does not escape in the specified max_iterations, then None is returned.
    Otherwise, a number in range(max_iterations) will be returned which tells us how many iterations passed before the point escaped.
    """

    z = 0 + 0j #initialize z as 0
    for i in range(max_iterations):
        z = z*z + c
        if abs(z) > 2:
            return i #return iteration count
    return None #return None if it does not escape


def get_complex_grid(
    top_left: complex,
    bottom_right: complex,
    step: float
) -> np.ndarray:
    """
    :param top_left: Top-left complex number (included in the grid)
    :param bottom_right: Bottom-right complex number (not included in the grid)
    :param step: The spacing in between points in the real and imaginary direction

    :return: A 2D array with complex numbers evenly spaced between top_left and bottom_right
    """

    real_range = np.arange(top_left + 0j, bottom_right + 0j, step) #array of real values
    imag_range = np.arange(top_left * 1j, bottom_right * 1j, -step) #array of imaginary values
    return real_range[:, None] + imag_range #combined array
