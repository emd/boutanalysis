'''Create 1D and 2D profiles to be used in BOUT++ simulations'''


import numpy as np
from scipy import interpolate

from boututils import file_import
from boutanalysis import grid


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


def interpolate2grid(prof, psi, grid_path, dim=2, pf='decreasing'):
    '''Interpolate given profile onto the specified BOUT++ grid.

    Parameters:
    prof -- 1D array, containing the profile (Q) to interpolate
        onto the BOUT++ grid. The profile should be a flux function
        such that Q = Q(psi)
    psi -- 1D array, containing the normalized poloidal flux
        (i.e. 0 on the magnetic axis and 1 on the separatrix)
    grid_path -- str, path to the grid file to interpolate onto
    dim -- int, valid options are: 1, 2
        dim specifies dimension of resulting interpolated profile;
        for example, many of the BOUT++ equilibrium profiles
        (e.g. pressure, jpar, etc) are specified as 2D grids.
        Dimensions of 3 or higher (which are *not* needed for
        axisymmetric equilibria) are not supported.
    pf -- str, valid options are: 'decreasing', 'flat', 'none'.
        As the flux decreases moving radially outwards in
        the private flux (PF) region, a simple flux function formalism is
        no longer valid in this region. Instead, we must determine
        an appropriate model for the PF region.

        --  'decreasing':
            Create a dummy flux variable, psi', defined as
                        psi' = 1 + |psi - 1|
            and then use the mapping Q_{PF}(psi') = Q_{non-PF}
            such that the profile in the PF region behaves
            similarly to the profile in the SOL

        --  'flat':
            Gives a flat profile in the PF region, with the
            constant value determined by the value of the
            profile on the separatrix

        --  'none':
            Simply uses the flux function formalism, i.e. Q = Q(psi).
            For equilibrium values, such as the pressure, this
            will give an unphysical grid and is *NOT* recommended

    Returns:
    An array of the speficied dimension interpolated onto the
    BOUT++ grid. This array can subsequently be written to
    the grid file for use in simulations.

    '''
    g = file_import(grid_path)

    if dim == 1:
        psi_grid = grid.grid2psi(g, vector=True)
        prof_interp = interpolate.spline(psi, prof, psi_grid)

    # TODO: Generalize this to double null
    elif dim == 2:
        prof_interp = np.zeros(g['Rxy'].shape)


       # PF region:

        # Determine the poloidal indices of the PF region
        pf_ind1 = np.arange(0, (g['jyseps1_1'] + 1))
        pf_ind2 = np.arange((g['jyseps2_2'] + 1), prof_interp.shape[1])
        pol_ind = np.concatenate((pf_ind1, pf_ind2))

        # Restricting ourselves to the poloidal PF domain identified above,
        # the PF region is fully specified by radial indices where psi < 1
        psi_grid = grid.grid2psi(g, vector=True, yind=pol_ind[0])
        rad_ind = np.where(psi_grid < 1.0)
        sep_ind = np.max(rad_ind) + 1

        if pf == 'decreasing':
            psi_dummy = 1.0 + np.abs(psi_grid[rad_ind] - 1)
            prof_interp[0:sep_ind, pol_ind] = interpolate.spline(
                    psi, prof, psi_dummy)[:, np.newaxis]
        elif pf == 'flat':
            prof_interp[0:sep_ind, pol_ind] = interpolate.spline(
                    psi, prof, 1.0)
        elif pf == 'none':
            prof_interp[0:sep_ind, pol_ind] = interpolate.spline(
                    psi, prof, psi_grid[rad_ind])[:, np.newaxis]


        # Non-PF region 1:

        # This region lies in the poloidal PF domain identified above,
        # but it does *not* satisfy psi < 1 (that is, this region is
        # in the SOL)
        rad_ind = np.where(psi_grid >= 1.0)
        prof_interp[sep_ind:, pol_ind] = interpolate.spline(
                psi, prof, psi_grid[rad_ind])[:, np.newaxis]


        # Non-PF region 2:

        # The entire radial domain in this region (core and SOL)
        # are *not* in the PF region
        psi_grid = grid.grid2psi(g, vector=True)
        pol_ind = np.arange(g['jyseps1_1'] + 1, g['jyseps2_2'])
        prof_interp[:, pol_ind] = interpolate.spline(
                psi, prof, psi_grid)[:, np.newaxis]

    else:
        raise ValueError('Interpolation not supported above 2D')

    return prof_interp
