"""pygabor.spectrogram 

Functions for computing spectrograms based on either the time-causal 
analogue of the Gabor transform, the discrete analogue of the Gabor 
transform or the sampled Gabor transform.

For the time-causal analogue of the Gabor transform, the temporal 
window function is chosen as the time-causal limit kernel, which 
constitutes a canonical temporal smoothing kernel for obtaining 
temporal scale covariance over a time-causal temporal domain.

For the discrete analogue of the Gabor transform, the temporal window 
function is chosen as the discrete analogue of the Gaussian kernel, 
which constitutes a canonical way to discretize the continuous 
Gaussian kernel over a non-causal discrete domain.

For the sampled Gabor transform, the temporal window function is 
chosen as the sampled Gaussian kernel, which may constitute the 
most commonly used approach for discretizing the continuous 
Gaussian kernel over a non-causal discrete domain, without 
further thoughts.

In the code below, there is a structure of handling default parameters
for the different methods based on class objects, and with an
associated structure for setting default values of these,
for more convenient use of the function for computing spectrograms.

Note! This code is a reimplementation, and not as thoroughly
tested as the original Matlab code used for the experiments
in the scientific publications.

References:

Lindeberg (2023) "A time-causal and time-recursive analogue of 
the Gabor transform", arXiv preprint arXiv:2308.14512.

Lindeberg (2023) "A time-causal and time-recursive scale-covariant 
scale-space representation of temporal signals and past time", 
Biological Cybernetics, 117(1-2): 21-59. (Contains extensive treatment 
of the time-causal limit kernel, with its relation to defining 
a temporal scale-space representation over multiple temporal scales, 
over a time-causal temporal domain.)

Lindeberg (2023) "Discrete approximations of Gaussian smoothing 
and Gaussian derivatives", arXiv preprint arXiv:2311.11317 
(Contains extensive treatment of the topic of approximating the 
continuous Gaussian kernel over a non-causal discrete domain.)

Lindeberg (1990) "Scale space for discrete signals", 
IEEE Transactions on Pattern Analysis and Machine Intelligence 
12(3): 234--254. (Contains original definition of the discrete 
analogue of the Gaussian kernel, constituting a canonical way 
to approximate the continuous Gaussian kernel over a non-causal 
discrete domain.)
"""

from typing import NamedTuple, Union
from math import sqrt, log2, ceil, pi
import numpy as np
from matplotlib import pyplot as plt

from pygabor.timecausgabor import timecausgabor
from pygabor.discgabor import discgabor


class TimeCausGaborMethod(NamedTuple):
    """Object for storing the parameters of a time-frequency-analysis 
method, based on the time-causal analogue of the Gabor transform.

The parameter c is the distribution parameter for the time-causal
limit kernel used as the temporal window function in the time-causal
frequency transform, where the parameter numlevels describes the
number of first-order recursive filters that are coupled in cascade
in the discrete approximation of the time-causal limit kernel.
"""
    name : str       # 'timecausgabor'
    c : float        # 2.0
    numlevels: int   # 8


def timecausgabormethodobject(
        c : float = 2.0,
        numlevels: int = 8
    ) -> TimeCausGaborMethod :
    """Creates an object that contains the parameters of a method object 
for the time-causal analogue of the Gabor transform, with default values 
for a preferred choice.
"""
    return TimeCausGaborMethod('timecausgabor', c, numlevels)


class DiscGaborMethod(NamedTuple):
    """Object for storing the parameters of a time-frequency-analysis 
method, based on the discrete analogue of the Gabor transform, obtained 
by choosing the temporal window function as the discrete analogue of 
the Gaussian kernel.

The parameter epsilon is the truncation error for the discrete approximation
of the Gaussian temporal window function truncated at the tails.
"""
    name: str       # 'discgabor'
    epsilon : float # 0.001


def discgabormethodobject(
        epsilon : float = 0.001
    ) -> DiscGaborMethod :
    """Creates an object that contains the parameters of a method object 
for the discrete analogue of the Gabor transform, with default values 
for a preferred choice.
"""
    return DiscGaborMethod('discgabor', epsilon)


class SamplGaborMethod(NamedTuple):
    """Object for storing the parameters of a time-frequency-analysis 
method, based on choosing the temporal window function as the sampled 
Gaussian kernel

The parameter epsilon represents an upper bound on the truncation error 
for the discrete approximation of the Gaussian temporal window function
truncated at the tails.
"""
    name: str       # 'samplgabor'
    epsilon : float # 0.001


def samplgabormethodobject(
        epsilon : float = 0.001
    ) -> SamplGaborMethod :
    """Creates an object that contains the parameters of a method 
object for the sampled Gabor transform, with default values for a 
preferred choice.
"""
    return SamplGaborMethod('samplgabor', epsilon)


def defaultimefreqanalmethod(
        method : str = 'timecausgabor-c2'
    ) -> Union[TimeCausGaborMethod, DiscGaborMethod, SamplGaborMethod] :
    """Converts a user-friendly string for naming a time-frequency 
analysis method into a method object
"""
    if method == 'timecausgabor-c2':
        return timecausgabormethodobject(2.0)

    if method == 'timecausgabor-csqrt2':
        return timecausgabormethodobject(sqrt(2.0))

    if method == 'discgabor':
        return discgabormethodobject()

    if method == 'samplgabor':
        return samplgabormethodobject()

    raise ValueError(f'Unknown time-frequency-analysis method {method}')


class TempWdwPars(NamedTuple):
    """Object for storing parameters for computing the temporal standard 
deviations of the temporal window functions from the frequencies, in the 
case when temporal standard deviations are not specified.

The parameter N represents the number of wavelengths for a given
frequency that are to be multiplied when setting the temporal standard
deviation of the temporal window function used in the time-frequency
transform. The parameters minstddev and maxstddev constitute soft
lower and upper bounds on the temporal standard deviation of the temporal
window function, to prevent too short temporal integration times at
high frequencies and to prevent too long temporal integration times
at lower frequencies. The parameter p is a power used for combining
the linearily predicted temporal standard deviation with the soft
lower and upper bounds on the temporal standard deviation.
"""
    N : int
    minstddev : float
    maxstddev : float
    p : float


def tempwdwparsobject(
        N : float = 8.0,
        minstddev : float = 0.001,
        maxstddev : float = 0.040,
        p : float = 2.0
    ) -> TempWdwPars :
    """Creates an object that contains the parameters of a method object 
for the time-causal analogue of the Gabor transform, with default values 
for a preferred choice.
"""
    return TempWdwPars(N, minstddev, maxstddev, p)


class Spectrogram(NamedTuple):
    """Object for storing a spectrogram as well as its attributes
"""
    method : Union[TimeCausGaborMethod, DiscGaborMethod, SamplGaborMethod]
    frequencies : np.ndarray
    stddevs : np.ndarray
    wdwpars : TempWdwPars
    coscomp : np.ndarray
    sincomp : np.ndarray
    samplfreq : float
    duration : float


def auditory_frequencies(
        fmin : float = 20,
        fmax : float = 16000,
        numperoctave : int = 48
    ) -> np.ndarray :
    """Computes a set of logarithmically distributed frequencies, 
between fmin and fmax, and using a given number of frequencies 
per octave
"""
    logfmin = log2(fmin)
    logfmax = log2(fmax)

    numfreqs = ceil(numperoctave*(logfmax - logfmin)) + 1
    logfreqs = np.linspace(logfmin, logfmax, numfreqs)

    return np.power(2, logfreqs)


def auditory_stddevs_from_frequencies(
        frequencies : np.ndarray,
        tmpwdwpars : TempWdwPars
    ) -> np.ndarray :
    """Computes temporal standard deviations from the frequencies, 
in such a way that the standard deviations are essentially proportional 
to the wavelength, but with soft thresholding in the lower and upper ends 
of the frequency range, to prevent too long and too short temporal 
integration times for the time-frequency transform
"""
    N = tmpwdwpars.N
    minstddev = tmpwdwpars.minstddev
    maxstddev = tmpwdwpars.maxstddev
    p = tmpwdwpars.p

    stddevs = ((minstddev**(p) + (N/frequencies)**(p))**(1/p))

    return stddevs/((1 + (stddevs/maxstddev)**(p))**(1/p))


def auditory_spectrogram(
        signal : np.ndarray,
        samplfreq : float,
        method : Union[str, \
                       TimeCausGaborMethod, DiscGaborMethod, SamplGaborMethod],
        frequencies : np.ndarray = None,
        stddevs : np.ndarray = None,
        wdwpars : TempWdwPars = None
    ) -> Spectrogram :
    """Computes an auditory spectrogram of a signal, with sampling
frequency samplfreq, given a specification of a time-frequency analysis 
method and a set of (regular not angular) frequencies and 
frequency-dependent temporal scales for the temporal window functions.

The argument method for the time-frequency analysis can be either of 
the following user-friendly short strings, which imply default setting
of the complementary parameters of the methods.

   'samplgabor' - the sampled Gabor transform
   'discgabor' - the discrete analogue of the Gabor transform
   'timecausgabor-c2' - the time-causal analogue of the Gabor transform
                        for distribution parameter c = 2
   'timecausgabor-csqrt2' - the time-causal analogue of the Gabor transform
                            for distribution parameter c = sqrt(2)

Alternatively, if you want to specify other values than the default
values for a time-frequency analysis method, you can instead provide a 
method object of the appropriate type.

If the array with the frequencies is None, then a default set of 
frequencies suitable for auditory analysis will be chosen.

If the array with temporal standard deviations for the temporal window 
functions is None, then a default set of temporal standard deviations 
will be chosen, as specified by the object wdwpars, and essentially 
proportional to the wavelengths of the frequencies, while with soft 
thresholding at the lower and higher ranges, to prevent either too long
temporal integration times for lower frequencies or too short temporal 
integration times for higher frequencies, when computing the spectrogram.
"""
    # Convert a possibly given user-friendly input string to a method object
    if isinstance(method, str):
        method = defaultimefreqanalmethod(method)

    # Issue warning if temporal standard deviations are specified,
    # but not frequencies, so that the latter will be set as default
    if stddevs is not None and frequencies is None:
        print('Warning! The standard deviations of the temporal window \
functions have been specified, but not any frequencies. Was this really \
your intention, to combine user set standard deviations with default \
frequencies set by the code?')

    # Set default frequencies if not specified
    if frequencies is None:
        frequencies = auditory_frequencies()

    # If both stddevs and wdwpars are set, avoid future confusions in the
    # output object, by setting the functionally not used wdwpars to None
    if stddevs is not None:
        wdwpars = None

    # Set default parameters for computing the temporal window functions
    if wdwpars is None and stddevs is None:
        wdwpars = tempwdwparsobject()

    # Compute temporal standard deviations, unless specified
    if stddevs is None:
        stddevs = \
          auditory_stddevs_from_frequencies(frequencies, wdwpars)

    # Ensure that all the arrays are 1-D arrays
    assert len(signal.shape) == 1
    assert len(frequencies.shape) == 1
    assert len(stddevs.shape) == 1

    # Allocate the output spectrogram and compute each row in it
    coscomp = np.zeros((frequencies.size, signal.size))
    sincomp = np.zeros((frequencies.size, signal.size))
    for i, frequency in np.ndenumerate(frequencies):
        if method.name == 'samplgabor':
            thisrow_cos, thisrow_sin = \
              discgabor(signal, 1/samplfreq, 2*pi*frequency, \
                        stddevs[i]*samplfreq, 'samplgauss', method.epsilon)
        elif method.name == 'discgabor':
            thisrow_cos, thisrow_sin = \
              discgabor(signal, 1/samplfreq, 2*pi*frequency, \
                        stddevs[i]*samplfreq, 'discgauss', method.epsilon)
        elif method.name == 'timecausgabor':
            thisrow_cos, thisrow_sin = \
              timecausgabor(signal, 1/samplfreq, 2*pi*frequency, \
                            stddevs[i]*samplfreq, method.c, method.numlevels)
        else:
            raise ValueError(f'Unknown time-frequency transform {method.name}')

        coscomp[i, :] = thisrow_cos[:]
        sincomp[i, :] = thisrow_sin[:]

    return Spectrogram(method, frequencies, stddevs, wdwpars, \
                       coscomp, sincomp, samplfreq, len(signal)/samplfreq)


def showlog(
        spectrogram : Spectrogram,
        lowsoftthresh : float = 0.000001,
        maxrange : float = 60):
    """Computes the logarithm of the absolute value of a spectrogram 
and displays it
"""
    # Compute the absolute value of the spectrogram
    absspectrogram = np.sqrt(spectrogram.coscomp**2 + spectrogram.sincomp**2)

    # Compute logarithmic magnitudes in dB, with additional lower bound
    maxval = np.max(absspectrogram)
    logspectrogram = 20 * np.log10(absspectrogram/maxval + lowsoftthresh)
    logspectrogram[logspectrogram < -maxrange] = -maxrange

    midi = midifromfreq(spectrogram.frequencies)

    im = \
      plt.imshow(logspectrogram, \
                 cmap='jet', interpolation='nearest', aspect='auto', \
                 origin='lower', \
                 extent=[0, spectrogram.duration, min(midi), max(midi)])
    plt.colorbar(im)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Log Frequency (MIDI semitones)")
    plt.show()


def harmonictestsignal(
        f0 : float = 440,
        duration : float = 0.3,
        numovertones : int = 7,
        samplfreq = 48000
    ) -> (np.ndarray, float) :
    """Creates a test signal with a harmonic spectrum
"""
    length = round(duration * samplfreq)
    outsignal = np.zeros(length)

    dt = 1/samplfreq
    time = dt * np.linspace(0, length-1, length)

    for i in range(numovertones):
        thistone = np.sin(2 * pi * f0 * (i + 1) * time)
        outsignal = outsignal + thistone / (i + 1)

    return outsignal, samplfreq


def midifromfreq(frequencies : np.ndarray) -> np.ndarray:
    """Converts regular frequencies in Hz to MIDI semitones
"""
    return 69 + 12 * np.log2(frequencies / 440)
