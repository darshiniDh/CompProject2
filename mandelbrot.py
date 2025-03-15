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


def get_escape_time_color_arr(c_arr: np.array, max_iterations: int) -> np.array:
    #Initialize escape_time array with the maximum value(num_iterations + 1for non-escaping points)
    escape_time = np.ones(c_arr.shape, dtype=int) * (max_iterations + 1)

    #Create arrays for real and imaginary parts of z, initialized to 0
    z_real = np.zeros_like(c_arr, dtype=float)
    z_imag = np.zeros_like(c_arr, dtype=float)

    #Iterate for each point in c_arr
    for iteration in range(max_iterations):
        #Compute Mandelbrot iteration: z = z^2 + c
        z_real_sq = z_real * z_real - z_imag * z_imag
        z_imag_sq = 2 * z_real * z_imag

        z_real = z_real_sq + c_arr.real # z^2 + c
        z_imag = z_imag_sq + c_arr.imag # z^2 + c

        #Find points where z > 2 to escape
        escape_mask = z_real**2 + z_imag**2 > 4

        #Set escape times for points
        escape_time[escape_mask] = np.minimum(escape_time[escape_mask], iteration + 1)

        #Stop iterating if all points escaped
        if np.all(escape_time <= max_iterations):
            break

        # Calculate color values using escape times
        color_arr = (max_iterations - escape_time + 1) / (max_iterations + 1)

        #Points that never escape are colored black (0.0)
        color_arr[escape_time == (max_iterations + 1)] = 0.0

    
def julia_set(c: complex, width: int, height: int, zoom: float, max_iterations: int) -> np.ndarray:
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

