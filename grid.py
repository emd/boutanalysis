'''Interact with BOUT++ grids, particularly those generated via Hypnotoad.'''


def grid2psi(g, vector = False, yind = None):
    '''Calculate the normalized flux coordinate psi from BOUT++ grid file g.

    Parameters:
    g -- a BOUT++ grid file, e.g. as generated from hypnotoad
    vector -- boolean; outputs 1d array psi(x) if true, 2d psi(x,y) if False
    yind -- integer; if returning a vector, gives y-index for slice

    Returns:
    Normalized flux coordinate psi corresponding to the BOUT++ grid file g,
    where "noramlized" means psi = 0 on axis and psi = 1 @ the separatrix.
    If vector == True, psi will be a 1d array psi(x); otherwise, psi will be
    a 2d array psi(x,y).

    '''
    psi = (g['psixy'] - g['psi_axis']) / (g['psi_bndry'] - g['psi_axis'])
    if vector:
        if yind is None:
            # Take slice in middle of y domain, avoiding PF regions
            # NOTE: This avoids PF regions only for single null
            yind = psi.shape[1] / 2
        return psi[:, yind]
    else:
        return psi


def grid2q(g):
    '''Compute the q profile from a BOUT++ grid file.

    Parameters:
    g -- the BOUT++ grid file, i.e. as computed from Hypnotoad

    Returns:
    the q profile with the radial resolution of the input grid file

    '''
    import numpy as np

    # ShiftAngle is the change in *toroidal* angle per poloidal transit
    return np.abs(g['ShiftAngle']) / (2 * np.pi)


def grid_path(input_file):
    '''Return the path to a grid file given the path to the BOUT.inp file.

    Parameters:
    input_file -- string, path to the BOUT.inp file

    Returns:
    path to the grid file

    NOTE: Assumes the path is the FIRST item denoted with 'grid' in BOUT.inp
    and also assumes it is of the form
                              grid = "..."
    Otherwise, this will fail... there may be a better way than this... 

    '''
    for line in open(input_file):
        if 'grid' in line:
            return str.split(line, '"')[1]

