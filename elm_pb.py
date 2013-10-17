'''Analyze results from the BOUT++ elm_pb 3-field module.'''


def omega_star_i(g, n, kappa, a, norm = True,
                 Z = 1, A = 2, density_norm = 1e20):
    '''Calculate the ion diamagnetic frequency for a given equilibrium.

    Parameters:
    g -- a BOUT++ grid file, e.g. as generated from Hypnotoad
    n -- scalar, the toroidal mode number
    kappa -- scalar, the plasma elongation, defined as b = kappa * a,
        where b is the semi-major axis and a is the semi-minor axis
        (i.e. a is the plasma minor radius)
    a -- scalar, the plasma minor radius; units = [m]
    norm -- boolean, normalize the ion diamagnetic frequency to the
        BOUT++ Alfven frequency if True; use physical units if False
    Z -- integer, the ion atomic number (i.e. number of protons)
    A -- integer, the ion mass number (i.e. m_i = A * m_p)
    denisty_norm -- scalar, the particle density normalization
        used in BOUT.inp

    Returns:
    the ion diamagnetic frequency, with units dictated by norm

    NOTE: The following assumptions are made to simplify the calculation:
    - The ion diamagnetic velocity is taken purely in the *poloidal*
      direction, corresponding to B_T >> B_P
    - P_i = P_e = P / 2
    - m = n * q, approximately...
    - The *maximal* radius R is located on the outboard midplane

    '''
    import numpy as np

    from boututils import file_import

    # compute the q-profile
    # ShiftAngle is the change in toroidal angle per poloidal transit
    q = np.abs(g['ShiftAngle']) / (2 * np.pi)

    # the poloidal wavenumber
    # units: [k_theta] = m^{-1}
    k_theta = n * q / (np.sqrt(kappa) * a)

    # Now, compute the ion diamagnetic velocity, V_D = (dp/dr) / (ni * q * B)
    # units: [V_D] = m/s
    #
    # First, determine the location of the outboard midplane.
    # It is assumed the outboard midplane shares the *same*
    # poloidal (y) index as the grid point with *maximum* R.
    #
    ind = np.where(g['Rxy'] == np.max(g['Rxy']))[1]
    p = 0.5 * g['pressure'][:, ind] # [p] = N/m^2, p_i = p_e = p / 2
    R = g['Rxy'][:, ind] # [R] = m
    B = g['Bxy'][:, ind] # [B] = T

    # Remove the degenerate dimension from p, R, and B to make truly 1d arrays
    p = np.squeeze(p)
    R = np.squeeze(R)
    B = np.squeeze(B)

    gradp = np.gradient(p) / np.gradient(R)
    ni20 = np.mean(g['Ni0']) # [ni20] = 10^20 m^{-3}

    # NOTE: ni * q = (ni20 * 1e20) * (Z * 1.6e-19) = 16 * Z * ni20
    V_D = np.abs(gradp) / (16 * Z * ni20 * B)

    omega = k_theta * V_D
    if norm:
        omega_A = v_A(g, density_norm, A) / g['rmag']
        return omega / omega_A
    else:
        return omega


def v_A(g, density_norm = 1e20, A = 2):
    '''Compute nominal Alfven velocity for a given BOUT++ elm_pb simulation.

    Parameters:
    g -- a BOUT++ grid file, e.g. as generated from Hypnotoad
    density_norm -- scalar, the main ion particle density normalization
        given in BOUT.inp
    A -- integer, the main ion mass number (i.e. m_i = A * m_p)

    Returns:
    the Alfven velocity v_A; [v_A] = m / s

    '''
    import numpy as np

    mu0 = 4 * np.pi * 1e-7 # [mu0] = SI units
    mp = 1.67e-27 # [mp] = kg
    B = g['bmag'] # [B] = T

    return B / np.sqrt(mu0 * density_norm * (A * mp))

