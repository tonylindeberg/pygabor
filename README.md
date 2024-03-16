# pygabor : Gabor transform toolbox for Python

Contains the following modules for performing time-frequency analysis on discrete signals:

## timecausgabor.py: Primitives for computing the time-causal analogue of the Gabor transform

The time-causal analogue of the Gabor transform is obtained by choosing the temporal window function in a time-frequency analysis as the time-causal limit kernel, which provides a theoretically well-founded way to define a time-frequency analysis over a time-causal temporal domain, for which the future cannot be accessed.

Notably, while the current code is written for offline experiments, the underlying approach is fully time-causal and time-recursive, and should therefore be highly suitable for real-time applications.

For examples of how to apply these functions for computing time-causal time-frequency transforms, please see the enclosed Jupyter notebook 
[timecausgabordemo.ipynb](https://github.com/tonylindeberg/pygabor/blob/main/timecausgabordemo.ipynb).


## discgabor.py: Primitives for computing the discrete analogue of the Gabor transform

The discrete analogue of the Gabor transform is obtained by choosing the temporal window function in a time-frequency analysis as the discrete analogue of the Gaussain kernel, which provides a theoretically well-founded way to discretize the continuous Gaussian kernel in such a way that the underlying theoretical properties, that make the Gaussian kernel a canonical choice over a non-causal temporal domain, do also hold after the discretization

For reference purpose, this package also provides a possibility to choose the temporal window function as the sampled Gaussian kernel, which may otherwise constitute the most commonly used choice for discretizing the continuous Gaussian kernel

For examples of how to apply these functions for computing non-causal time-frequency transforms, please see the enclosed Jupyter notebook 
[discgabordemo.ipynb](https://github.com/tonylindeberg/pygabor/blob/main/discgabordemo.ipynb).


## spectrogram.py: Functions for computing spectrograms based on the above time-frequency analysis methods

Computes combined spectrograms, by applying either the time-causal analogue of the Gabor transform or the discrete analogue of the Gabor transform over a set of logarithmically distributed frequencies.

The current focus of code is with regard to auditory spectrograms. With minor modifications, the approach is, however, also applicable to computing spectrograms for other application domains.

For examples of how to use these functions for computing different types of time-causal or non-causal spectrograms, please see the enclosed Jupyter notebook 
[spectrogramdemo.ipynb](https://github.com/tonylindeberg/pygabor/blob/main/spectrogramdemo.ipynb).


## Installation:

This package is available 
through pip and can installed by

```bash
pip install pygabor
```

This package can also be downloaded directly from GitHub:

```bash
git clone git@github.com:tonylindeberg/pygabor.git
```


## Dependencies:

This package depends on the 
[pyscsp](https://github.com/tonylindeberg/pyscsp)
and 
[pytempscsp](https://github.com/tonylindeberg/pytempscsp)
packages, for performing the temporal smoothing in the time-frequency transforms. 

The pyscsp and pytempscsp packages are available at PyPi and at GitHub:

```bash
pip install pyscsp
```

```bash
git clone git@github.com:tonylindeberg/pyscsp.git
```

```bash
pip install pytempscsp
```

```bash
git clone git@github.com:tonylindeberg/pytempscsp.git
```

## References:

Lindeberg (2023) "A time-causal and time-recursive analogue of the Gabor transform", arXiv preprint arXiv:2308.14512.
([Preprint](https://arxiv.org/abs/2308.14512))
(Defines and describes the time-causal analogue of the Gabor transform, based on using the time-causal limit kernel as the temporal window function for performing time-frequency analysis. A discrete analogue of the Gabor transform is also defined, by discretizing the continuous Gaussian kernel in the regular Gabor transform using the discrete analogue of the Gaussian kernel.)

Lindeberg (2023) "A time-causal and time-recursive scale-covariant scale-space representation of temporal signals and past time", Biological Cybernetics, 117(1-2): 21-59. 
([Open Access](http://dx.doi.org/10.1007/s00422-022-00953-6))
(Contains an extensive treatment of the time-causal limit kernel, with its relation to defining a temporal scale-space representation at multiple temporal scales, over a time-causal temporal domain.)

Lindeberg (2023) "Discrete approximations of Gaussian smoothing and Gaussian derivatives", arXiv preprint arXiv:2311.11317
([Preprint](https://arxiv.org/abs/2311.11317))
(Contains an extensive treatment of the topic of discretizing the continuous Gaussian kernel over a non-causal discrete domain.)

Lindeberg (1990) "Scale space for discrete signals", IEEE Transactions on Pattern Analysis and Machine Intelligence 12(3): 234--254.
([Preprint](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A472968&dswid=6991))
(Contains the original definition of the discrete analogue of the Gaussian kernel, which constitutes a canonical choice to discretize the continuous Gaussian kernel over a non-causal discrete domain, in such a way that the properties that make the continuous Gaussian kernel special over a continuous domain do also hold after the discretization.)



