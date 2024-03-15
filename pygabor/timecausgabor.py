"""Functions for computing discrete approximations of the time-causal 
analogue of the Gabor transform.

Note! This code is a reimplementation, and not as thoroughly tested as 
the original Matlab code used for the experiments in the scientific 
publications.

Reference:

Lindeberg (2023) "A time-causal and time-recursive analogue of 
the Gabor transform", arXiv preprint arXiv:2308.14512.
"""

from typing import Union
import numpy as np
from pytempscsp.tempscsp import limitkernfilt, limitkernfilt_mult


def timecausgabor(
        signal : np.ndarray,
        time: Union[float, np.ndarray],
        omega : float,
        sigma : float,
        c : float = 2.0,
        numlevels: int = 8,
        method: str = 'explicitcascade'
) -> (np.ndarray, np.ndarray) :
    """Computes one slice of a discrete approximation of the time-causal 
analogue of the Gabor transform of a signal for a given angular frequency omega 
and using temporal smoothing with temporal standard deviation sigma.

The argument time may be either a scalar float specifying the temporal
sampling distance delta_t or a (uniformly spaced) time vector containing
the actual time values associated with the values of the signal.

The argument c is the distribution parameter for the time-causal limit
kernel used for temporal smoothing, and the argument numlevels represents
the number of layers by which this kernel is approximated using a set
of discrete first-order recursive filters coupled in cascade.

The output from this function is a pair of arrays, representing the
real and imaginary parts of the transform for the given angular
frequency omega.
"""
    if signal.ndim != 1:
        raise ValueError('The input signal must be one-dimensional')

    if isinstance(time, float):
        delta_t = time
        time = delta_t * np.linspace(0, signal.size-1, signal.size)

    omegat = omega * time
    fcoswt = signal * np.cos(omegat)
    fsinwt = - signal * np.sin(omegat)

    filtcoswt = limitkernfilt(fcoswt, sigma, c, numlevels, method)
    filtsinwt = limitkernfilt(fsinwt, sigma, c, numlevels, method)

    return filtcoswt, filtsinwt


def timecausgabor_mult(
        signal : np.ndarray,
        time: Union[float, np.ndarray],
        omega : float,
        sigmamin : float,
        sigmamax : float,
        c : float = 2.0,
        numlevels: int = 8
) -> (np.ndarray, np.ndarray, np.ndarray) :
    """Computes one slice of a discrete approximation of the time-causal 
analogue of the Gabor transform of a signal for a given angular frequency omega 
and using temporal smoothing with a set of temporal standard deviations
between sigmamin and sigmamax (and possibly extended because of the
ratio between successive scale levels as given by the scale ratio c).

By providing a range of temporal standard deviations, time-frequency
analysis representations over multiple temporal scales can be computed
more efficiently, compared to using multiple calls of the above function
timecausgabor() for multiple values of the temporal standard deviations
of the temporal window functions in the time-causal time-frequency
transform.

The argument time may be either a scalar float specifying the temporal
sampling distance delta_t or a (uniformly spaced) time vector containing
the actual time values associated with the values of the signal.

The argument c is the distribution parameter for the time-causal limit
kernel used for temporal smoothing, and the argument numlevels represents
the number of layers by which this kernel is approximated using a set
of discrete first-order recursive filters coupled in cascade.

The output from this function is a pair of arrays, representing the
real and imaginary parts of the transform for the given angular
frequency omega at each scale, complemented by an array with the values of
the scale parameters sigma being used.
"""
    if signal.ndim != 1:
        raise ValueError('The input signal must be one-dimensional')

    if isinstance(time, float):
        delta_t = time
        time = delta_t * np.linspace(0, signal.size-1, signal.size)

    omegat = omega * time
    fcoswt = signal * np.cos(omegat)
    fsinwt = - signal * np.sin(omegat)

    filtcoswt, sigmavec = \
      limitkernfilt_mult(fcoswt, sigmamin, sigmamax, c, numlevels)
    filtsinwt, sigmavec = \
      limitkernfilt_mult(fsinwt, sigmamin, sigmamax, c, numlevels)

    return filtcoswt, filtsinwt, sigmavec
