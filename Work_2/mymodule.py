import numpy as np

def rmsd(poly_coeff, x_values, y_values):
    """Calculate the root mean square deviation of a polynomial approximation.

    Args:
    poly_coeff (ndarray): Coefficients of the polynomial.
    x_values (ndarray): x-coordinates of the data points.
    y_values (ndarray): y-coordinates of the data points.

    Returns:
    float: The root mean square deviation value.
    """
    poly_values = np.polyval(poly_coeff, x_values)
    return np.sqrt(np.mean((poly_values - y_values) ** 2))