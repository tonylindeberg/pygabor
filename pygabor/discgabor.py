"""Functions for computing discrete approximations of the Gabor transform.

Note! This code is a reimplementation, and not as thoroughly tested as 
the original Matlab code used for the experiments in the scientific 
publications.

Reference:

Lindeberg (2023) "A time-causal and time-recursive analogue of 
the Gabor transform", arXiv preprint arXiv:2308.14512.
"""

from typing import Union, List
import numpy as np
from pyscsp.discscsp import scspconv, scspconv_mult, ScSpMethod


def discgabor(
        signal : np.ndarray,
        time: Union[float, np.ndarray],
        omega : float,
        sigma : float,
        scspmethod : Union[str, ScSpMethod] = 'discgauss',
        epsilon : float = 0.00000001
) -> (np.ndarray, np.ndarray) :
    """Computes one slice of a discrete approximation of the Gabor
transform of a signal for a given angular frequency omega 
and using temporal smoothing with temporal standard deviation sigma.

The argument time may be either a scalar float specifying the temporal
sampling distance delta_t or a (uniformly spaced) time vector containing
the actual time values associated with the values of the signal.

The argument scspmethod specifies the discretization method to use to
approximate the continuous Gaussian smoothing operation, and the 
argument epsilon an upper bound on the truncation error.

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

    filtcoswt = scspconv(fcoswt, sigma, scspmethod, epsilon)
    filtsinwt = scspconv(fsinwt, sigma, scspmethod, epsilon)

    return filtcoswt, filtsinwt


def discgabor_mult(
        signal : np.ndarray,
        time: Union[float, np.ndarray],
        omega : float,
        sigmavec : List[float],
        scspmethod : Union[str, ScSpMethod] = 'discgauss',
        epsilon : float = 0.00000001
) -> (np.ndarray, np.ndarray) :
    """Computes one slice of a discrete approximation of the Gabor
transform of a signal for a given angular frequency omega 
and using temporal smoothing with a set of temporal standard deviations
as specified by the list sigmavec, and which are to be ordered in
increasing order. Compared to using multiple calls to the function
discgabor() for different values of sigma, this function uses
the cascade smoothing property of the Gaussian kernel to reduce
the computational work for smoothing over time, when computing
discrete approximations of the Gabor transform over multiple scales.

The argument time may be either a scalar float specifying the temporal
sampling distance delta_t or a (uniformly spaced) time vector containing
the actual time values associated with the values of the signal.

The argument scspmethod specifies the discretization method to use to
approximate the continuous Gaussian smoothing operation, and the 
argument epsilon an upper bound on the truncation error.

The output from this function is a pair of arrays, representing the
real and imaginary parts of the transform for the given angular
frequency omega at each scale.
"""
    if signal.ndim != 1:
        raise ValueError('The input signal must be one-dimensional')

    if isinstance(time, float):
        delta_t = time
        time = delta_t * np.linspace(0, signal.size-1, signal.size)

    omegat = omega * time
    fcoswt = signal * np.cos(omegat)
    fsinwt = - signal * np.sin(omegat)

    filtcoswt = scspconv_mult(fcoswt, sigmavec, scspmethod, epsilon)
    filtsinwt = scspconv_mult(fsinwt, sigmavec, scspmethod, epsilon)

    return filtcoswt, filtsinwt


def whitenoisesignal1D(length: int) -> np.ndarray:
    """Generates a while noise signal for testing purposes"""
    signal = np.random.normal(0.0, 1.0, length)
    return signal
