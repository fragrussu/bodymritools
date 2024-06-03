This script can be called directly from the command line. Its input parameters are the following:

```
usage: getT2ivimkurtReg.py [-h] [--mask <file>] [--sigma <file>] [--nsa <value>] [--ncpu <value>] [--grid <value>] [--pmin <list>] [--pmax <list>] [--wtk <value>] dwi scheme out

Non-linear least square fitting with offset-Gaussian noise model and Tikhonov regularisation of the two-pool T2-IVIM-Kurtosis model to diffusion MRI images obtained by varying b-value b and echo time TE. This provides voxel-
wise stimates of s0 (apparent proton density), fv (vascular signal fraction), Dv (vascular apparent diffusion coefficient), T2v (vascular T2), Dt (tissue apparent diffusion coefficient), Kt (tissue apparent excess kurtosis
coefficient), T2t (tissue T2), so that the signal model is s(b,TE) = s0*(fv*exp(-bDv -TE/T2v) + (1-fv)*exp(-bDt +(Kt/6)*b^2*Dt^2 -TE/T2t )). Note that this code can be used even if the TE was not varied during acquisition:
in this latter case, a very short "fake" TE value (e.g., 1ms) should be indicated for all MRI measurements, and both T2v and T2t should be fixed to a very long "fake" value (e.g., 1000ms) using options --pmin and --pmax (see
below). Dependencies: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel. Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain (<fgrussu@vhio.net>)

positional arguments:
  dwi             4D Nifti file of magnitude images from a diffusion MRI experiment obtained by varying b-value b and echo time TE.
  scheme          text file storing b-values and echo times used to acquired the images (space-separated elements; 1st row: b-values in s/mm^2; 2nd row: echo times TE in ms). If all images were acquired with the same echo
                  time (so that vascular/tissue T2 cannot be modelled), enter a very short TE for all measurement (e.g., 1 ms) and fix both vascular and tissue T2 to a very long value (for instance, 1000 ms) using the --pmin
                  and --pmax options below.
  out             root name for output files, to which the following termination strings will be added: "*_s0.nii" for the apparent proton density map; "*_fv.nii" for the vascular signal fraction map; "*_dv.nii" for the
                  vascular apparent diffusion coefficient map (in um2/ms); "*_t2v.nii" for the vascular T2 map (in ms); "*_dt.nii" for the tissue apparent diffusion coefficien map (in um2/ms); "*_kt.nii" for the tissue
                  apparent excess kurtosis map; "*_t2t.nii" for the tissue T2 (in ms); "*_fobj.nii" for the fitting objective function; "*_exit.nii" for the fitting exit code (1: success; -1: warning; 0: background);
                  "*_nfloor.nii" for the noise floor map (if a noise map sigma was provided with option --sigma)

options:
  -h, --help      show this help message and exit
  --mask <file>   3D Nifti storing the tissue mask, so that voxels storing 1 indicate that fitting is required, while voxels storing 0 indicate that fitting is not required
  --sigma <file>  3D Nifti storing a voxel-wise noise standard deviation map, which will be used to model the noise floor in the offset-Gaussian fitting objective function
  --nsa <value>   number of signal averages (optional, and relevant only if a noise standard deviation map is provided with option --sigma; default: 1. Note that in some vendors it is known as NSA or NEX, i.e. number of
                  excitations. Remember that trace-DW imaging has an intrinsic number of signal averages of 3, since images for three mutually orthogonal gradients are acquired and averaged)
  --ncpu <value>  number of threads to be used for computation (default: all available threads)
  --grid <value>  number of values for each tissue parameter in grid search (default: 3; be aware that computation time grows considerably as this parameter increases)
  --pmin <list>   comma-separaterd list storing the lower bounds for tissue parameters s0, fv, Dv (um2/ms), T2v (ms), Dt (um2/ms), Kt, T2t (ms). Example: --pmin 0.6,0.1,10.0,200.0,0.3,-0.5,50.0 (default:
                  0.8,0.0,4.0,150.0,0.2,0.0,20.0). Note: the true s0 bound will be obtained multiplying the value passed here by max(signal).
  --pmax <list>   comma-separaterd list storing the upper bounds for tissue parameters s0, fv, Dv (um2/ms), T2v (ms), Dt (um2/ms), Kt, T2t (ms). Example: --pmax 3.5,1.0,80.0,600.0,2.8,3.0,150.0 (default:
                  12.0,1.0,100.0,250.0,3.0,5.0,140.0). Note: the true s0 bound will be obtained multiplying the value passed here by max(signal).
  --wtk <value>   Tikhonov regularisation weight used in the fitting objective function (default 0.0; if set to 0.0, no regularisation is performed. Note that an effective regularisation weight value will depend on the range
                  on which the MRI signal is defined: for instance, for an MRI signal varying in the range [0; SMAX], resonable regularisation weights are of the order of approx SMAX/3)

```
