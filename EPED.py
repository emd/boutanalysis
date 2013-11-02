'''Compare BOUT++ results to those from EPED1.6'''

import numpy as np
from scipy import interpolate

import boutanalysis as ba

def omega_star_eff(g, n, kappa, a, Z = 1, A = 2, density_norm = 1e20):
    '''Compute the effective diamagnetic stabilization using EPED1.6 model.

    Parameters:
    g -- a BOUT++ grid file, e.g. as computed from Hypnotoad
    n -- numpy array, the toroidal mode number(s)
    kappa -- scalar, the plasma elongation, defined as b = kappa * a,
        where b is the semi-major axis and a is the semi-minor axis
        (i.e. a is the plasma minor radius)
    a -- scalar, the plasma minor radius; units = [m]
    Z -- integer, the ion atomic number (i.e. number of protons)
    A -- integer, the ion mass number (i.e. m_i = A * m_p)
    denisty_norm -- scalar, the particle density normalization
        used in BOUT.inp

    Returns:
    The effective diamagnetic stabilization frequency, omega_star_eff.
    The units of omega_star_eff are omega_A = v_A / R, which is
    identical to the frequency normalization used in BOUT++'s elm_pb
    module.

    omega_star_eff is computed via a bilinear fit to BOUT++ results as
    described in

        Snyder et al. Nucl. Fusion 51 (2011) 103016.

    The relevant fitting information is given in Fig. 4 of the article
    and the surrounding text. In particular, there is a region of strong
    diamagnetic stabilization (region 1) for low n, and a region of weak
    diamagnetic stabilization (region 2) for high n. The transition
    between the two regions occurs at a critical value of n given by

        n_crit = 27.7 / q_95

    This is the same model of diamagnetic stabilization used in EPED1.6.

    '''
    omega_star_i = ba.elm_pb.omega_star_i(g, n, kappa, a, Z = Z, A = A,  
                                          density_norm = density_norm) 
    omega_star_i = np.amax(omega_star_i, axis = 1) 

    psi = ba.grid.grid2psi(g, vector = True)
    q = ba.grid.grid2q(g)

    # Determine value of q at psi = 0.95 by doing a spline fit and evaluating.
    # However, q has a sharp kink at psi > 1, so we will only perform the
    # spline fit for psi < 1.
    ind = np.where(psi < 1)
    tck = interpolate.splrep(psi[ind], q[ind])
    q_95 = interpolate.splev(0.95, tck)

    n_crit = 27.7 / q_95

    omega = np.zeros(len(n))

    for i in range(len(n)):
        if n[i] < n_crit:
            omega[i] = np.sqrt(0.5) * omega_star_i[i] 
        else:
            omega[i] = np.sqrt(0.5) * (n_crit + 0.168 * (n - n_crit)) * omega_star_i[i] / n[i]

    return omega
