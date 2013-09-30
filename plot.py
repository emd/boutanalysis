'''Plot useful BOUT++ results quickly and easily.'''


def contour_over_psi0(s, g, RZ = False):
    '''Create contour plot of signal s over the equilibrium flux surfaces.

    Parameters:
    s -- 2d signal s(x,y)
    g -- BOUT++ grid file, e.g. as created by hypnotoad
    RZ -- boolean; real space (R,Z) if true, in field-aligned (x,y) if False

    Returns:
    ???? I'm sure this could be modified to return somehting useful...

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    from boutanalysis import grid

    # Compute *equilibrium* normalized flux coordinates
    psi0 = grid.grid2psi(g)

    # Number of psi contours. Ideas is to have a contour for every 0.01
    nLev = int(np.round((np.max(psi0) - np.min(psi0)) * 100))

    fig = plt.figure()
    if RZ:
        ax = fig.add_subplot(111)
        Cpsi0 = ax.contour(g['Rxy'], g['Zxy'], psi0, nLev, colors = 'k')
        plt.clabel(Cpsi0, inline = 1)
        Cs = ax.contourf(g['Rxy'], g['Zxy'], s, 30)
        plt.colorbar(Cs)
        ax.set_aspect('equal')
    else:
        Cpsi0 = plt.contour(psi0, nLev, colors = 'k')
        plt.clabel(Cpsi0, inline = 1)
        Cs = plt.contourf(s, 30)
        plt.colorbar(Cs)
    plt.show()
