'''Tools for interacting with VARYPED model equilibria'''

import numpy as np
from sys import argv
import string
import copy


def create_db(file_path):
    '''Create a dictionary from a VARYPED results text file.

    Parameters:
    file_path -- string, path to the text file containing VARYPED results.
        The first line of the file should begin with "VARYPED RESULTS...",
        the second line should give the column titles, e.g. i, pres, cur, 
        and the remaining lines should give the model equilibria values.

    Returns:
    a dictionary that will be keyed by the VARYPED model equlibria index 
    with values that are themselves dictionaries of the parameters of 
    the corresponding model equilibria.
 
    '''
    f = open(file_path)

    # v will be keyed by the model equlibria index, and its values will be 
    # dictionaries of the parameters of the corresponding model equilibria 
    v = dict()
    
    # Create a temporary dictionary 
    d = dict()

    # Read the lines from f into the dictionary v 
    for linenum, line in enumerate(f):
        if linenum == 1:
            # The titles of each column will be used as dictionary keys
            keys = line.split()
        elif linenum > 1:
            for i, val in enumerate(line.split()):
                if keys[i] == 'i':
                    d[keys[i]] = int(val)
                else:
                    d[keys[i]] = float(val)
            v[d['i']] = copy.copy(d)
    f.close()

    return v


def find_eq(v, p = None, J = None):
    '''Find VARYPED model equlibria indices with the specified parameters.

    Parameters:
    v -- dictionary, the keys will be the VARYPED model equilibria 
        indices, and the value pairs will themselves be dictionaries
        of the parameters of the corresponding equilibria
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


def get_params(v, ind):
    '''Get model parameters for specified VARYPED model equilibria.

    Parameters:
    v -- dictionary, the keys will be the VARYPED model equilibria 
        indices, and the value pairs will themselves be dictionaries
        of the parameters of the corresponding equilibria
    ind -- list, the index of equilibria to obtain parameters for

    Returns:
    params -- list, with each line giving an ordered pair (p, J), where
        p is the fraction of the experimental pressure gradient and
        J is the fraction of the experimental edge current density. 
    '''
    params = list()

    for i in ind:
        params.append((v[i]['pres'], v[i]['cur']))

    return params 


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

    v = create_db(file_path)
    ind = find_eq(v, p, J)
    params = get_params(v, ind)    

    if ind is not None:
        column = '{:<6}'
        print '\nVARYPED Equilibria:'
        print column.format('i') + column.format('p') + column.format('J')
        print '-'*15
        for i in range(len(ind)): 
            out = (column.format(str(ind[i]))
                + column.format(str(params[i][0])) 
                + column.format(str(params[i][1])))
            print out
        print '' 
