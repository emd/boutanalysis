'''Tools for interacting with VARYPED model equilibria'''

import numpy as np
from sys import argv
import string
import copy


def find_eq(file_path, p = None, J = None):
    '''Find VARYPED model equlibria indices with the specified parameters.

    Parameters:
    file_path -- string, path to the text file containing the VARYPED results.
        The first line of the file should begin with "VARYPED RESULTS...",
        the second line should give the column titles, e.g. i, pres, cur, 
        and the remaining lines should give the model equilibria values.
    p -- scalar, specifying the fraction of the experimental pressure 
        gradient. To find the VARYPED model equilibria indices with 
        75% of the experimental pressure gradient, specify p = 0.75, etc
    J -- scalar, specifying the fraction of the experimental edge current
        density. To find the VARYPED model equilibria indices with 
        110% of the experimental edge current density, specify J = 1.10

    Returns:
    The indices of the VARYPED equilibria with the specified pressure
    gradients and edge current densities.
 
    '''
    if p is None and J is None:
        print 'No parameters specified. Returning NoneType object.'
        return None

    f = open(file_path)

    # v will be keyed by the model equlibria index, and its values will be 
    # dictionaries of the parameters of the corresponding model equilibria 
    v = dict()
    
    # Create a temporary dictionary 
    d = dict()

    # Read the lines from f into the dictionary v 
    for linenum, line in enumerate(f):
        if linenum == 1:
            # The titles of each column will be used as dictionary keys later
            keys = line.split()
        elif linenum > 1:
            for i, val in enumerate(line.split()):
                d[keys[i]] = float(val)
            v[d['i']] = copy.copy(d)
    f.close()

    # Sort through the dictionary v for model equilibria that have the 
    # specified parameters and add their index to the list ind.
    # NOTE: We also only count equilibria that converged (convrg = 1),
    # as unconverged equilibria are useless to us.
    ind = list()
    for eq in v.keys():
        if p is None:
            if v[eq]['cur'] == J and v[eq]['convrg'] == 1:
                ind.append(eq)
        elif J is None:
            if v[eq]['pres'] == p and v[eq]['convrg'] == 1:
                ind.append(eq)
        elif v[eq]['pres'] == p and v[eq]['cur'] == J and v[eq]['convrg'] == 1:
            ind.append(eq) 
    
    return ind


if __name__ == '__main__':
    file_path, p, J = argv[1:4]

    if p == 'None':
        p = None
    else:
        p = float(p)
    if J == 'None':
        J = None
    else:
        J = float(J)

    ind = find_eq(file_path, p, J)

    if ind is not None:
        print 'VARYPED Equilibria:'
        for i in ind: 
            print '\t' + str(i)     
