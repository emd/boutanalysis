'''Analyze BOUT++ output signals.'''

import numpy as np
import matplotlib.pyplot as plt


class RandomData:
    def __init__(self, s, dt=1, axis_t=0, axis_ensemble=-1):
        # Initialize the Sf array to the proper dimensions
        if axis_ensemble == -1:
            shape = s.shape[:-1]
        else:
            shape = s.shape[:axis] + s.shape[(axis + 1):]
        self.Sf = np.zeros(shape, dtype=complex)

        # Take the ensemble averaged FFT in time
        for sample in np.arange(s.shape[axis_ensemble]):
            self.Sf += np.fft.fft(np.rollaxis(s, axis_ensemble)[sample,:],
                    axis=axis_t)
        self.Sf = self.Sf / s.shape[axis_ensemble]
        self.Sf = np.fft.fftshift(self.Sf, axes=axis_ensemble)

        # Obtain the corresponding frequencies
        self.f = np.fft.fftfreq(s.shape[axis_t], d=dt)
        self.f = np.fft.fftshift(self.f)

        # Obtain the phase
        self.phase = np.angle(self.Sf)


class KzSpec:
    '''Spectra, growth rate, and frequency for each k_z Fourier mode.

    Parameters:
    -----------
    s -- array, signal to be Fourier analyzed. It is assumed that
        time is the first coordinate (e.g. s = s(t,...)) and that
        z is the last coordinate (e.g. s = s(...,z))
    dz -- scalar, the spacing between adjacent z-grid points;
        [dz] = fractional torus circumference, that is

                dz = 2 * pi / (ZPERIOD * (MZ - 1))

        where ( 1 / ZPERIOD ) is the fraction of the torus simulated.
        dz is computed by BOUT++ and can be read directly from
        a BOUT++ dump file.
    t -- array, the time array corresponding to the values in s(t).
        If the user does *not* assign a value to t, a uniformly-spaced
        time array with dt=1 is assumed.

    Attributes:
    -----------
    sk -- array, of the same dimensions as the signal s(t,...,z),
        giving the spatial Fourier transform (in z) of the signal
    n -- array, the toroidal mode numbers corresponding to each
        k_z mode such that sk = sk(t,...,n)
    gammak -- array, of the same dimensions as the signal s(t,...,z),
        giving the instantaneous local growth rate.
        [gammak] = 1 / [t]
    freqk -- array, of the same dimensions as the signal s(t,...,z),
        giving the instantaneous local rotation frequency
        in the z-direction; [freqk] = 1 / [t]

    '''
    def __init__(self, s, dz=1, t=None):
        # Fourier transform in z, where z is *last* coordinate in signal
        self.sk = np.fft.fft(s)

        # The toroidal mode number, n
        self.n = 2 * np.pi * np.fft.fftfreq(s.shape[-1], d=dz)

        if t is None:
            t = np.arange(s.shape[0])

        # In order to broadcast the dt array with the signal array,
        # we must add a dummy axis to the dt array for every spatial
        # dimension of the signal array such that the dimensions
        # of the two arrays are commensurate
        dt = np.diff(t)
        for axis in np.arange(len(s.shape[1:])):
            dt = dt[:, np.newaxis]

        # Determine the
        #   (1) growth rate (gammak) and
        #   (2) rotation frequency in the z-direction (freqk)
        # for each k_z Fourier mode
        lambdak = np.diff(np.log(self.sk), axis=0) / dt
        self.gammak = np.real(lambdak)
        self.freqk = np.imag(lambdak)


def moment_xyzt(s):
    '''Calculate the moments of a 4d signal (t,x,y,z).

    Parameters:
    s -- 4d signal (t,x,y,z), as obtained via s = collect('s')

    Returns:
    dictionary, {'rms' = s_rms, 'dc' = s_dc, 'ac' = s_ac}, where
        (1) s_rms: 3d signal (t,x,y) that gives the
            root mean square of s in the z (toroidal) direction
        (2) s_dc: 3d signal (t,x,y) that gives the
            average of s in the z (toroidal) direction
        (3) s_ac: 4d signal (t,x,y,z) that gives the
            z (torodial) fluctuations of s

    NOTE: One can reconstruct the full signal via s = s_dc + s_ac

    '''
    # Take the RMS of the signal
    s_rms = np.sqrt(np.mean((s ** 2), axis = 3))

    # Take the average in the z-direction
    s_dc = np.mean(s, axis = 3)

    # Subtract the average to obtain the fluctuating z-component
    s_ac = s - s_dc[:, :, :, np.newaxis] # newaxis to broadcast s_dc 3d into 4d

    return {'rms': s_rms, 'dc': s_dc, 'ac': s_ac}


def growth_rate(s, t = None, plot = True):
    '''Compute the growth rate of a 3d signal s = s(t,x,y).

    Parameters:
    s -- a 3d signal s(t,x,y)
    t -- a 1d array with time values corresponding to the signal s
    plot -- boolean, plots growth rate vs time if True

    Returns:
    gamma -- a 1d array of growth rate [\omega_A] as a function of time

    '''
    # Calculate the max value at each point in time
    s = np.amax(np.amax(s, axis = 1), axis = 1)

    # The first element of s is often zero. This gives an
    # annoying 'divide by zero' warning when taking the log(),
    # so we will remove the first element of the s and t arrays.
    s = s[1:-1]
    if t is not None:
        t = t[1:-1]

    # Calculate the growth rate as a function of time
    if t is None:
        # Assumes dt = 1 Alfven time
        gamma = np.diff(np.log(s))
    else:
        # t should have the SAME length as the t-dimension of s
        if t.shape[0] != s.shape[0]:
            raise TypeError, "Time and signal arrays have incompatible dimensions"
        else:
            # Allows for nonuniform dt
            gamma = np.diff(np.log(s)) / np.diff(t)

    # Plot the growth rate vs time
    if plot:
        if t is None:
            plt.plot(gamma)
        else:
            plt.plot(t[0:-1], gamma)
        plt.xlabel('$t$ $[\\tau_A]$')
        plt.ylabel('$\gamma$ $[\omega_A]$')
        plt.show()

    return gamma


def omega(s, t, t0):
    '''Compute the angular frequency of a 1d signal s = s(t).

    Parameters:
    s -- a 1d signal s(t)
    t -- a 1d array with time values corresponding to the signal s
    t0 -- a scalar; time values below this value are ignored

    Returns:
    omega -- the angular frequency [\omega_A] of the signal s.

    The premise of this routine is to find the peaks in the signal
    (assumed to be varying as exp(i * omega * t)), and then 
    calculate the period T by realizing the distance between the
    extrema will be T / 2

    Future Improvements:
    (1) Somehow integrate with calculation of growth rate
    (2) Use something more accurate/elegant, e.g. Fourier Transforms
    (3) Handling of signals of many dimensions
    (4) The *sign* of the frequency (e.g. electron or ion direction)
    (5) Normalized vs physical units as an option 

    '''
    from scipy.signal import argrelextrema

    ## Plot signal so user can select appropriate range to find extrema
    #plt.plot(t, s)
    #plt.xlabel('$t$ [$\\tau_A$]')
    #plt.ylabel('Signal $s(x_0, t)$')
    #plt.show()

    # obtain minimum time, above which we will search for extrema,
    # and convert this minimum time into an appropriate signal index
    #t0 = float(raw_input('t0 = '))
    ind = np.where(np.abs(t - t0) == np.min(np.abs(t - t0)))
    t = t[ind[0]:]
    s = s[ind[0]:]

    maxind = argrelextrema(s, np.greater)
    minind = argrelextrema(s, np.less)

    tmax = t[maxind]
    tmin = t[minind]

    dt = np.abs(tmax[-1] - tmin[-1])
    T = 2 * dt

    return 2 * np.pi / T 
