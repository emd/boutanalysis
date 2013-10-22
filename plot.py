'''Plot useful BOUT++ results quickly and easily.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
    
from boutanalysis import grid


def contour_over_psi0(s, g, RZ = False, show = False):
    '''Create contour plot of signal s over the equilibrium flux surfaces.

    Parameters:
    s -- 2d signal s(x,y)
    g -- BOUT++ grid file, e.g. as created by hypnotoad
    RZ -- boolean; real space (R,Z) if true, in field-aligned (x,y) if False
    show -- boolean; show the resulting figure; when batch processing,
        set to False and save returned figure fo later viewing, but
        during an interactive session, set to True
 
    Returns:
    The figure object fig. This can be displayed later via plt.show(fig)
    or saved for later viewing via fig.savefig('foo.pdf').

    '''
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
    
    if show:
        plt.show()
    
    return fig


def growth_rate(gamma, t = None, show = False):
    '''Plot the growth rate gamma vs time t.

    Parameters:
    gamma -- array; a 1d signal gamma(t) giving growth rate as a 
        function of time
    t -- array; the time basis corresponding to gamma; if this 
        array is *not* specified, it will be assumed that 
        the time difference between successive points is one Alven time 
    show -- boolean; show the resulting figure; when batch processing,
        set to False and save returned figure fo later viewing, but
        during an interactive session, set to True
 
    Returns:
    The figure object fig. This can be displayed later via plt.show(fig)
    or saved for later viewing via fig.savefig('foo.pdf').
    '''
    fig = plt.figure()
    
    if t is None:
        plt.plot(gamma)
    else:
        # gamma is computed via np.diff(), and thus is reduced in length
        # by one relative to the time array
        plt.plot(t[1:-2], gamma)
    plt.xlabel('$t$ $[\\tau_A]$')
    plt.ylabel('$\gamma$ $[\omega_A]$')
    
    if show:
        plt.show()
  
    return fig


def eps_p(g, interp = False):
    '''Plot the expansion parameter eps_p = rho_s / L_n.'''
    psi0 = grid.grid2psi(g, vector = True)
    eps_p = grid.eps_p(g)
   
    # Does not work yet... 
    #if interp:
    #    tck = interpolate.splrep(psi0, eps_p)
    #    psi0 = np.linspace(np.min(psi0), np.max(psi0), 250)
    #    eps_p = interpolate.splev(psi0, tck)   

    plt.plot(psi0, eps_p)
    plt.xlabel('$\psi$')
    plt.ylabel('$\epsilon_p = \\rho_s / L_p$')
    plt.show()    

