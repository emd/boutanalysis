'''Interact with BOUT++ grids, particularly those generated via Hypnotoad.'''

import numpy as np
from scipy import interpolate

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
    #import numpy as np

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


def outboard_midplane_index(g):
    '''Return the *poloidal* index for the outboard midplane of a BOUT++ grid.

    Parameters:
    g -- the BOUT++ grid file, i.e. as computed from Hypnotoad

    Returns:
    The poloidal index of the outboard midplane 

    NOTE: Assumes the outboard midplane occurs where the plasma 
    achieves its *largest* major radius coordinate.

    '''
    return np.where(g['Rxy'] == np.max(g['Rxy']))[1]   


def outboard_midplane_LCFS(g, units = 'cm'):
    '''Return array of distances from LCFS along outboard midplane.
    
    Parameters:
    g -- the BOUT++ grid file, i.e. as computed from Hypnotoad
    units -- string; unit of distance; available: m, cm, mm
   
    Returns:
    An array of distances from the Last Closed Flux Surface (LCFS), 
    or separatrix, along the outboard midplane in specified units. 
 
    ''' 
    psi0 = grid2psi(g, vector = True)
    polind = outboard_midplane_index(g)
    R = np.squeeze(g['Rxy'][:, polind]) # [R] = m
    
    tck = interpolate.splrep(psi0, R)
    Rsep = interpolate.splev(1.0, tck)
    dR = R - Rsep 

    if units == 'm':
        return dR
    elif units == 'cm':
        return 100 * dR
    elif units == 'mm':
        return 1000 * dR
    else:
        raise ValueError, "units can have values of 'm', 'cm', and 'mm'"  


def eps_p(g):
    '''Return the expansion parameter epsilon_p = rho_s / L_p.

    Parameters:
    g -- the BOUT++ grid file, i.e. as computed from Hypnotoad

    Returns:
    epsilon_p = rho_s / L_p, which is a common expansion parameter
        used in gyrokinetics and MHD. rho_s is the ion sound 
        gyroradius, while L_p is the pressure scale length.

    '''
    A = 2 # mass number of deuteron
    
    ind = outboard_midplane_index(g) 
    R = g['Rxy'][:, ind] # [R] = m 
    B = g['Bxy'][:, ind] # [B] = T
    Te = g['Te0'][:, ind] # [Te] = eV
    p = g['pressure'][:, ind] # [p] = Pascals
    
    R = np.squeeze(R)
    B = np.squeeze(B)
    Te = np.squeeze(Te)
    p = np.squeeze(p)

    rho_s = 1.02e-4 * (np.sqrt(A * Te) / B) # [rho_s] = m

    grad_p = np.abs(np.gradient(p) / np.gradient(R)) 
    
    # Find where the pressure has zero gradient (such as in the SOL);
    # these regions can cause divide by zero errors in the L_p calculation
    ind1 = np.where(grad_p != 0)
    ind2 = np.where(grad_p == 0)
 
    L_p = np.zeros(len(p))
    L_p[ind1] = p[ind1] / grad_p[ind1]
    L_p[ind2] = None

    return rho_s / L_p 
