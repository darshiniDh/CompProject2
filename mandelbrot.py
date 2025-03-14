import numpy as np

def get_escape_time(c: complex, max_iterations: int) -> int | None:
    """
    :param c: Complex number
    :param max_iterations: Specified number of iterations

    :return: The number of iterations which pass before c escapes.
    If c does not escape in the specified max_iterations, then None is returned.
    Otherwise, a number in range(max_iterations) will be returned which tells us how many iterations passed before the point escaped.
    """

    z = c #initialize z as c
    if abs(z) > 2:
        return 0
    for i in range(1, max_iterations+1):
        z = z**2 + c
        if abs(z) > 2:
            return i #return iteration count
    return None #return None if it does not escape


def get_complex_grid(
    top_left: complex,
    bottom_right: complex,
    step: float
) -> np.ndarray:
    """
    Generates a 2D array of complex numbers evenly spaced between top_left and bottom_right.

    :param top_left: Top-left complex number (included in the grid)
    :param bottom_right: Bottom-right complex number (not included in the grid)
    :param step: The spacing in between points in the real and imaginary direction

    :return: A 2D array with complex numbers evenly spaced between top_left and bottom_right
    """

    # Extract real and imaginary parts separately
    real_values = np.arange(top_left.real, bottom_right.real, step)  # Correct real range
    imag_values = np.arange(top_left.imag, bottom_right.imag, -step)  # Correct imaginary range (descending)

    # Create a 2D grid of complex numbers
    return real_values + 1j * imag_values[:, None]  # Broadcasting to form a complex grid
