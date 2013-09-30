'''Compare BOUT++ results to those from EPED1.6'''


def omega_star_eff(g, n):
    '''Compute the effective diamagnetic stabilization using EPED1.6 model.

    Parameters:
    g -- a BOUT++ grid file, e.g. as computed from Hypnotoad
    n -- array, the toroidal mode number

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
    import numpy as np
    from scipy import interpolate

    from boutanalysis import grid 

    # slopes of regions 1 and 2
    m1 = 0.0411764285714
    m2 = 0.00678730769231

    psi = grid.grid2psi(g, vector = True)
    q = grid.grid2q(g)

    # Determine value of q at psi = 0.95 by doing a spline fit and evaluating.
    # However, q has a sharp kink at psi > 1, so we will only perform the
    # spline fit for psi < 1.
    ind = np.where(psi < 1)
    tck = interpolate.splrep(psi[ind], q[ind])
    q_95 = interpolate.splev(0.95, tck)

    n_crit = 27.7 / q_95

    omega = np.zeros(len(n))
    omega_crit = m1 * n_crit

    for i in range(len(n)):
        if n[i] < n_crit:
            omega[i] = m1 * n[i]
        else:
            omega[i] = omega_crit + (m2 * (n[i] - n_crit))

    return omega


# Computes the slopes for the bilinear fitting function described in:
#
#        Snyder et al. Nucl. Fusion 51 (2011) 103016
#
# and prints the slopes to screen. 
if __name__ == '__main__':
    import numpy as np
    
    n = np.array([0, 14, 40])
    omega = np.array([0, 0.57647, 0.75294])
    
    m1 = (omega[1] - omega[0]) / (n[1] - n[0])
    m2 = (omega[2] - omega[1]) / (n[2] - n[1])
    
    print 'slope in region 1: m1 = ' + str(m1)
    print 'slope in region 2: m2 = ' + str(m2)
