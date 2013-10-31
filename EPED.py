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
    # The strength of the diamagnetic stabilization in regions 1 and 2
    # are *fractions* of the maximum ion diamagnetic stabilization. 
    # Further, we care about the strength of this stabilization as a 
    # function of the toroidal mode number, n. Thus, we want to 
    # determine the rate m_max at which the ion diamagnetic stabilization 
    # changes with n, e.g. m_max = d(omega_{*i,max})/dn. 
    # 
    # Now, as omega_{*i,max} ~ n, we can simply compute
    # m_max = (omega_{*i,max}(n=1) - omega_{*i,max}(n=0)) / (1 - 0)
    # i.e. m_max = omega_{*i,max}(n=1)
    omega_n1 = ba.elm_pb.omega_star_i(g, np.array([1]), kappa, a, Z = Z, 
                                      A = A, density_norm = density_norm) 
    m_max = np.amax(omega_n1, axis = 1) 
    
    # slopes of regions 1 and 2 
    m1 = 0.735026371695 * m_max 
    m2 = 0.121157912906 * m_max 

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
   
    # omega_{*eff} curve from Fig. 4 
    n = np.array([0, 14, 40])
    omega = np.array([0, 0.57647, 0.75294])
    
    m1 = (omega[1] - omega[0]) / (n[1] - n[0])
    m2 = (omega[2] - omega[1]) / (n[2] - n[1])
    
    # omega_{*max} curve from Fig. 4 
    n = np.array([0, 17.8352])
    omega = np.array([0, 0.999134])
    m_max = (omega[1] - omega[0]) / (n[1] - n[0])

    print 'slope in region 1: m1 = ' + str(m1 / m_max) + ' m_max'
    print 'slope in region 2: m2 = ' + str(m2 / m_max) + ' m_max'
