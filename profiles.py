'''Create 1D and 2D profiles to be used in BOUT++ simulations'''


import numpy as np
from scipy import interpolate


def csv_import(path, column, skip_header=1):
    '''Import a 1D profile from a CSV file.

    Parameters:
    path -- string, path to the file containing the 1D profile
    column -- int, if the CSV file contains several columns,
        specify the index of the column containing the desired
        profile, with 0 corresponding to the first column
    skip_header -- int, the number of header rows to skip
        before the profile information begins

    Returns:
    An array containing the 1D profile contained in the CSV file.

    '''
    return np.genfromtxt(open(path), usecols=column, skip_header=skip_header)
