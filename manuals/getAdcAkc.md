```
usage: getAdcAkc.py [-h] [--mask <file>] [--noise <file>] [--savg <num>] [--nw <num>] [--pmin <list>] [--pmax <list>] [--ncpu <num>]
                    [--sldim <num>] [--nlalgo <string>]
                    s_file bval_file out

Fit a phenomenological representation of the diffusion MRI signal acquired at varying b-values based on the apparent diffusion coefficient
(ADC) and apparent excess kurtosis coefficient (AKC). Third-party dependencies: nibabel, numpy, scipy. Author: Francesco Grussu, Vall d Hebron
Institute of Oncology (VHIO). Email: <fgrussu@vhio.net>.

positional arguments:
  s_file             path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values
  bval_file          path of a text file storing a list of b-values in s/mm2 (space-separated, in FSL format), with each element of this list
                     indicating the b-value of the corresponding diffusion-weighted volume in the input 4D NIFTI file
  out                root file name of output files; output NIFTIs will be stored as double-precision floating point images (FLOAT64), and the
                     file names will end in: *_ADC.nii (ADC map um2/ms), *_AKC.nii (excess kurtosis map), *_S0.nii (estimate of the MRI signal
                     at b = 0), *_exit.nii (voxel-wise exit code; -1: warning, error in model fitting; 0: background; 1 successful parameter
                     estimation). If a noise map was provided through the noisefile input parameter (see below), then additional output NIFTI
                     files storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood at maximum), *_BIC.nii (Bayesian
                     Information Criterion) and *_AIC.nii (Akaike Information Criterion)

options:
  -h, --help         show this help message and exit
  --mask <file>      3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
  --noise <file>     3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the expected Rician
                     noise floor.
  --savg <num>       number of signal averages used for MRI data acquisition (default: 1). This parameter is used for the estimation of the
                     noise floor (it is ignored if the option --noise is not used). Note that in some vendors, this parameter is also referred
                     to as number of excitations (NEX).
  --nw <num>         number of values to test for each unknown tissue parameter in the grid search (it must be an integer; default 10)
  --pmin <list>      comma-separaterd list of lower bounds for tissue parameters, in this order: ADC (um2/ms); Kurtosis excess AKC; non-DW
                     signal S0 (espressed in normalised form, i.e. the true non-DW signal divided by the maximum observed signal as S0 =
                     S0true/max(s)). Example: 0.2,0.0,0.9. Default: 0.1,0.0,0.5
  --pmax <list>      comma-separaterd list of upper bounds for tissue parameters, in this order: ADC (um2/ms); Kurtosis excess AKC; non-DW
                     signal S0 (espressed in normalised form, i.e. the true non-DW signal divided by the maximum observed signal as S0 =
                     S0true/max(s)). Example: 2.4,3.5,3.5. Default: 3.0,5.0,5.0
  --ncpu <num>       number of threads to be used for computation (default: 1, single thread)
  --sldim <num>      image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2,
                     implying parallel processing along the 3rd image dimension)
  --nlalgo <string>  algorithm to be used for constrained objective function minimisation in non-linear fitting. Choose among: "Nelder-Mead",
                     "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" (default: "trust-constr" - see documentation of
                     scipy.optimize.minimize for information on the optimisation algorithm)   
```
