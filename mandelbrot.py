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


def get_escape_time_color_arr(c_arr: np.ndarray, max_iterations: int) -> np.ndarray:
    """
    Compute the escape time color array for the Mandelbrot set.

    Parameters:
    c_arr (np.ndarray): A 2D array of complex numbers representing points in the complex plane.
    max_iterations (int): Maximum number of iterations to determine escape time.

    Returns:
    np.ndarray: A 2D array of the same shape as c_arr with greyscale color values in [0,1].
    """
    # Initialize escape_time array with the maximum value (max_iterations + 1 for non-escaping points)
    escape_time = np.full(c_arr.shape, max_iterations + 1, dtype=int)

    # Create arrays for real and imaginary parts of z, initialized to 0
    z_real = np.zeros_like(c_arr, dtype=float)
    z_imag = np.zeros_like(c_arr, dtype=float)

    # Iterate for each point in c_arr
    for iteration in range(1, max_iterations + 1):
        # Compute Mandelbrot iteration: z = z^2 + c
        z_real_sq = np.square(z_real) - np.square(z_imag)
        z_imag_sq = 2 * z_real * z_imag

        z_real = z_real_sq + c_arr.real  # z^2 + c
        z_imag = z_imag_sq + c_arr.imag  # z^2 + c

        # Find points where |z| > 2 to escape, using np.abs to avoid overflow warnings
        escaped = np.greater(np.square(z_real) + np.square(z_imag), 4, where=np.isfinite(z_real) & np.isfinite(z_imag))

        # Set escape times for newly escaped points
        escape_time[np.logical_and(escaped, escape_time == max_iterations + 1)] = iteration

        mask = np.abs(z_real) < 1e10
        z_real = np.where(mask, z_real, 1e10)
        z_imag = np.where(mask, z_imag, 1e10)

    color_arr = (max_iterations - escape_time + 1) / (max_iterations + 1)
    color_arr[escape_time == (max_iterations + 1)] = 0.0
    
    return color_arr

def get_julia_color_arr(c: complex, width: int, height: int, zoom: float, max_iterations: int) -> np.ndarray:
    #defines points 
    x = np.linspace(-2 / zoom, 2 / zoom, width)
    y = np.linspace(-2 / zoom, 2 / zoom, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    #creates escape time array
    escape_time = np.full(Z.shape, max_iterations, dtype=int)
    #Iteration
    for i in range(max_iterations):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + c
        escape_time[mask & (np.abs(Z) > 2)] = i

    return escape_time

