# Code for Numerical Experiments in "Efficient Forced Response Computations of Acoustical Systems with a State-Space Approach"
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5159476.svg)](https://doi.org/10.5281/zenodo.5159476)

This repository contains code for the numerical experiments reported in 

> Art J. R. Pelling, Ennes Sarradj
> **Efficient Forced Response Computations of Acoustical Systems with a State-Space Approach**
> Acoustics 2021, 3, 581-593.
> [doi:10.3390/acoustics3030037](https://doi.org/10.3390/acoustics3030037)

The input data are taken from the Aachen Multi-Channel Impulse Response Database (MIRD)[[1]](#1) and can be downloaded [here](https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_0.160s__3-3-3-8-3-3-3.zip "MIRD Database").

It contains an implementation of randomized SVD as described in [[2]](#2) on the basis of [this repository](https://github.com/gwgundersen/randomized-svd). Our implementation is less verbose and faster. Furthermore, fast matrix-vector multiplications for block-Hankel matrices are enabled by providing a dedicated `ndarray` subclass.

## Installation and Usage
The downloaded database archive has to be extracted into "./data/MIRD/" in order to be loaded correctly.

The required dependencies are listed in [requirements.txt](requirements.txt) and can be installed into a new conda envrionment with
```bash
$ cd /path/to/repo
$ conda create -n <env> -f=requirements.txt
```

After these steps, the figures can be created by setting the appropriate parameters in [script.py](script.py) and executing
```bash
$ conda activate <env>
$ python script.py
```

## References
<a id = "1">[1]</a>
E. Hadad, F. Heese, P. Vary and S. Gannot, "Multichannel audio database in various acoustic environments", 2014 14th International Workshop on Acoustic Signal Enhancement (IWAENC), 2014, pp. 313-317, doi: [10.1109/IWAENC.2014.6954309](https://doi.org/10.1109/IWAENC.2014.6954309).

<a id = "2">[2]</a>
N. Halko, P. G. Martinsson, and J. A. Tropp, "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions", SIAM Review 2011 53:2, 2011, 217-288, doi: [10.1137/090771806](https://doi.org/10.1137/090771806).
