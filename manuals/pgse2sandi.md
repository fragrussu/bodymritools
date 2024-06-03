This script can be called directly from the command line. Its input parameters are the following: 
```
usage: pgse2sandi.py [-h] [--mask <file>] [--noise <file>] [--nw <num>] [--reg <Lnorm,weight>] [--pmin <p1min,...,p7min>] [--pmax <p1max,...,p7max>] [--ncpu <num>] [--sldim <num>] [--nlalgo <string>] [--sm <flag>]
                     dwi scheme out

Regularised non-linear least square fitting of the SANDI model from Palombo et al, NeuroImage 2020, doi: 10.1016/j.neuroimage.2020.116835. Third-party dependencies: nibabel, numpy, scipy. Last successful test with
nibabel=5.1.0, scipy=1.10.1, numpy=1.23.5. Francesco Grussu, Vall d Hebron Institute of Oncology <fgrussu@vhio.net>. Copyright (c) 2024, Vall d Hebron Institute of Oncology (VHIO), Barcelona, Spain. All rights reserved.

positional arguments:
  dwi                   path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values and, potentially, multiple diffusion times
  scheme                path of a text file storing the sequence parameters as a space-separated text file, made of 3 rows x M columns, where column m = 1, ..., M corresponds to the m-th volume of the input NIFTI file. --
                        First row: b-values in s/mm2; -- Second row: gradient duration small delta in ms; -- Third row: gradient separation Large Delta in ms
  out                   root file name of output files; output NIFTIs will be stored as double-precision floating point images (FLOAT64), and the file names will end in: *_Dinum2ms.nii (intrinsic intra-neurite diffusivity in
                        um2/ms), *_Dsomaum2ms.nii (intrinsic soma diffusivity in um2/ms), *_Dexum2ms.nii (extra-neurite, extra-soma apparent diffusion coefficient in um2/ms), *_Rsomaum.nii (soma radius in um), *_Fin.nii
                        (intra-neurite signal fraction), *_Fex.nii (extra-neurite, extra-soma signal fraction), *_S0.nii (estimate of the MRI signal at b = 0), *_exit.nii (voxel-wise exit code; -1: warning, error in model
                        fitting; 0: background; 1 successful parameter estimation).

options:
  -h, --help            show this help message and exit
  --mask <file>         3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
  --noise <file>        3D noise standard deviation map in NIFTI format (used for noise floor modelling)
  --nw <num>            number of values to test for each unknown tissue parameter in the grid search (it must be an integer; default 5)
  --reg <Lnorm,weight>  comma-separated list of parameters for fitting regularisation specifying i) the type of L-norm (1 for LASSO, 2 for Tikhonov), ii) the weight of the regulariser, ranging in [0.0,1.0]. Default: 2,0.001
                        (L2 norm, with a weight of 0.001). Set 2,0.0 for a standard non-linear least square fitting with no regularisation.
  --pmin <p1min,...,p7min>
                        comma-separated list of lower bounds for tissue parameters, in this order: intrinsic intra-neurite diffusivity (um2/ms), intrinsic soma diffusivity (um2/ms), extra-neurite, extra-soma apparent
                        diffusion coefficient (um2/ms), soma radius (um), intra-neurite signal fraction, extra-neurite extra-soma signal fraction, and S0, described as non-DW signal normalised by the maximum observed signal
                        (S0 = S0true/max(signal)). Defalut are 0.5,2.99,0.5,1.0,0.01,0.01,0.95. If regularisation is used, avoid bounds that are exactly equal to 0.0.
  --pmax <p1max,...,p7max>
                        comma-separated list of upper bounds for tissue parameters, in this order: intrinsic intra-neurite diffusivity (um2/ms), intrinsic soma diffusivity (um2/ms), extra-neurite, extra-soma apparent
                        diffusion coefficient (um2/ms), soma radius (um), intra-neurite signal fraction, extra-neurite extra-soma signal fraction, and S0, described as non-DW signal normalised by the maximum observed signal
                        (S0 = S0true/max(signal)). Defalut are 3.0,3.0,3.0,15.0,0.99,0.99,1.05. If regularisation is used, avoid bounds that are exactly equal to 0.0.
  --ncpu <num>          number of threads to be used for computation (default: 1, single thread)
  --sldim <num>         image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)
  --nlalgo <string>     algorithm to be used for constrained objective function minimisation in non-linear fitting. Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" (default: "trust-
                        constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
  --sm <flag>           flag indicating whether the spherical mean signals should be saved or not. Set to 0 or 1 (defalut = 0, do not save spherical mean signals). If set to 1, the following two output files will be
                        produced: *_SphMean.nii: 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation; *_SphMean.acq.txt: space-separated text file storing the sequence parameters
                        correspnding to *_SphMean.nii. It features the same number of columns as *_SphMean.nii, and has 3 lines (first row: b-values in s/mm2; second row: gradient duration small delta in ms; third row:
                        gradient separation Large Delta in ms). Anything different from 0 will be treated as True (save spherical mean).
```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module pgse2sandi:

NAME
    pgse2sandi

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

FUNCTIONS
    run(mrifile, mriseq, output, maskfile=None, noisefile=None, Nword=5, regnorm=2, regw=0.001, pmin=[0.5, 2.99, 0.5, 1.0, 0.01, 0.01, 0.95], pmax=[3.0, 3.0, 3.0, 15.0, 0.99, 0.99, 1.05], nthread=1, slicedim=2, fitting='trust-constr', saveSM=False)
        Regularised non-linear least square fitting of the SANDI model from Palombo et al, NeuroImage 2020, doi: 10.1016/j.neuroimage.2020.116835 
        
        USAGE
        run(mrifile, mriseq, output, maskfile=None, noisefile=None, Nword=5, ...
                pmin=[0.5,2.99,0.5,1.0,0.01,0.01,0.95], pmax=[3.0,3.0,3.0,15.0,0.99,0.99,1.05], nthread=1, slicedim=2, fitting='trust-constr', saveSM=False)
        
        * mrifile:    path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values and/or diffusion times                       
                      
        * mriseq:     path of a text file storing the sequence parameters as a space-separated text file, made of 3 rows x M columns, 
                                      where column m = 1, ..., M corresponds to the m-th volume of the input NIFTI file. 
                                      -- First row: b-values in s/mm2 
                                      -- Second row: gradient duration small delta in ms
                                      -- Third row: gradient separation Large Delta in ms
                      
        * output:     root file name of output files; output NIFTIs will be stored as double-precision floating point images 
                      (FLOAT64), and the file names will end in:
                      *_Dinum2ms.nii (intrinsic intra-neurite diffusivity in um2/ms),
                      *_Dsomaum2ms.nii (intrinsic soma diffusivity in um2/ms),
                      *_Dexum2ms.nii (extra-neurite, extra-soma apparent diffusion coefficient in um2/ms),
                      *_Rsomaum.nii (soma radius in um),
                      *_Fin.nii (intra-neurite signal fraction),
                      *_Fex.nii (extra-neurite, extra-soma signal fraction), 
                      *_S0.nii (estimate of the MRI signal at b = 0), 
                      *_exit.nii (voxel-wise exit code; -1: warning, error in model fitting; 0: background; 
                                  1 successful parameter estimation).
                      
        * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
                      
        * noisefile:  3D noise standard deviation map in NIFTI format (used noise floor modelling) 
                      
        * Nword:      number of values to test for each tissue parameter grid search (default: 5)
        
        * regnorm:    type of L-norm for regularisation (1 for LASSO, 2 for Tikhonov; default: 2) 
            
        * regw:       weight of the regulariser (ranging in [0.0,1.0]; default: 0.001; set it to 0.0 for 
                      standard non-linear least square fitting with no regularisation)
        
        * pmin:       list of lower bounds for tissue parameters, in this order:
                      intrinsic intra-neurite diffusivity (um2/ms)
                      intrinsic soma diffusivity (um2/ms)
                      extra-neurite, extra-soma apparent diffusion coefficient (um2/ms)
                      soma radius (um)
                      intra-neurite signal fraction
                      extra-neurite, extra-soma signal fraction 
                      S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)) 
                      
                      Default values are [0.5,0.5,0.5,1.0,0.01,0.01,0.95]
                      If regularisation is used, avoid bounds that are exactly equal to 0.0
        
        * pmax:       list of upper bounds for tissue parameters, in this order:
                      intrinsic intra-neurite diffusivity (um2/ms)
                      intrinsic soma diffusivity (um2/ms)
                      extra-neurite, extra-soma apparent diffusion coefficient (um2/ms)
                      soma radius (um)
                      intra-neurite signal fraction
                      extra-neurite, extra-soma signal fraction 
                      S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)) 
                      
                      Default values are [3.0,3.0,3.0,12.0,0.99,0.99,1.05]
                      If regularisation is used, avoid bounds that are exactly equal to 0.0
        
        * nthread:    number of threads to be used for computation (default: 1, single thread)
        
        * slicedim:   image dimension along which parallel computation will be exectued when nthread > 1
                      (can be 0, 1, 2 -- default slicedim=2, implying parallel processing along 3rd 
                      image dimension)
        
        * fitting:   algorithm to be used for constrained objective function minimisation in non-linear fitting. 
                      Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" 
                      (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
        
        * saveSM:    boolean flag indicating whether the spherical mean signals should be saved or not. Defalut = False. If True,
                      If True, the following two output files will be produced:
                      *_SphMean.nii: 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation
                      *_SphMean.acq.txt: space-separated text file storing the sequence parameters correspnding to *_SphMean.nii. It features
                                         the same number of columns as *_SphMean.nii, and has 3 lines:
                                         -- First row: b-values in s/mm2 
                                         -- Second row: gradient duration small delta in ms
                                         -- Third row: gradient separation Large Delta in ms
                
        Third-party dependencies: nibabel, numpy, scipy
        Last successful test with nibabel=5.1.0, scipy=1.10.1, numpy=1.23.5
        
        Author: Francesco Grussu, Vall d'Hebron Institute of Oncology
                <fgrussu@vhio.net>
                        
        Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron 
        (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain).

DATA
    erf = <ufunc 'erf'>
        erf(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, um2/ms)
                      extra-neurite, extra-soma apparent diffusion coefficient (um2/ms)
                      soma radius (um)
                      intra-neurite signal fraction
                      extra-neurite, extra-soma signal fraction 
                      S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)) 
                      
                      Default values are [0.5,0.5,0.5,1.0,0.01,0.01,0.95]
                      If regularisation is used, avoid bounds that are exactly equal to 0.0
        
        * pmax:       list of upper bounds for tissue parameters, in this order:
                      intrinsic intra-neurite diffusivity (um2/ms)
                      intrinsic soma diffusivity (um2/ms)
                      extra-neurite, extra-soma apparent diffusion coefficient (um2/ms)
                      soma radius (um)
                      intra-neurite signal fraction
                      extra-neurite, extra-soma signal fraction 
                      S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)) 
                      
                      Default values are [3.0,3.0,3.0,12.0,0.99,0.99,1.05]
                      If regularisation is used, avoid bounds that are exactly equal to 0.0
        
        * nthread:    number of threads to be used for computation (default: 1, single thread)
        
        * slicedim:   image dimension along which parallel computation will be exectued when nthread > 1
                      (can be 0, 1, 2 -- default slicedim=2, implying parallel processing along 3rd 
                      image dimension)
        
        * fitting:   algorithm to be used for constrained objective function minimisation in non-linear fitting. 
                      Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" 
                      (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
        
        * saveSM:    boolean flag indicating whether the spherical mean signals should be saved or not. Defalut = False. If True,
                      If True, the following two output files will be produced:
                      *_SphMean.nii: 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation
                      *_SphMean.acq.txt: space-separated text file storing the sequence parameters correspnding to *_SphMean.nii. It features
                                         the same number of columns as *_SphMean.nii, and has 3 lines:
                                         -- First row: b-values in s/mm2 
                                         -- Second row: gradient duration small delta in ms
                                         -- Third row: gradient separation Large Delta in ms
                
        Third-party dependencies: nibabel, numpy, scipy
        Last successful test with nibabel=5.1.0, scipy=1.10.1, numpy=1.23.5
        
        Author: Francesco Grussu, Vall d'Hebron Institute of Oncology
                <fgrussu@vhio.net>
                        
        Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron 
        (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain).

DATA
    erf = <ufunc 'erf'>
        erf(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
        
        erf(z, out=None)
        
        Returns the error function of complex argument.
        
        It is defined as ``2/sqrt(pi)*integral(exp(-t**2), t=0..z)``.
        
        Parameters
        ----------
        x : ndarray
            Input array.
        out : ndarray, optional
            Optional output array for the function values
        
        Returns
        -------
        res : scalar or ndarray
            The values of the error function at the given points `x`.
        
        See Also
        --------
        erfc, erfinv, erfcinv, wofz, erfcx, erfi
        
        Notes
        -----
        The cumulative of the unit normal distribution is given by
        ``Phi(z) = 1/2[1 + erf(z/sqrt(2))]``.
        
        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Error_function
        .. [2] Milton Abramowitz and Irene A. Stegun, eds.
            Handbook of Mathematical Functions with Formulas,
            Graphs, and Mathematical Tables. New York: Dover,
            1972. http://www.math.sfu.ca/~cbm/aands/page_297.htm
        .. [3] Steven G. Johnson, Faddeeva W function implementation.
           http://ab-initio.mit.edu/Faddeeva
        
        Examples
        --------
        >>> import numpy as np
        >>> from scipy import special
        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(-3, 3)
        >>> plt.plot(x, special.erf(x))
        >>> plt.xlabel('$x$')
        >>> plt.ylabel('$erf(x)$')
        >>> plt.show()
```   
