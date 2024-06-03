This script can be called directly from the command line. Its input parameters are the following: 
```
usage: mri2micro_dictml.py [-h] [--mask <file>] [--noise <file>] [--savg <num>] [--nw <num>] [--pmin <list>] [--pmax <list>] [--reg <Lnorm,weight>] [--ncpu <num>] [--sldim <num>] [--nlalgo <string>] [--rgalgo <string>]
                           [--rgpars <list>]
                           s_file sig_dict par_dict out

This tool performs maximum-likelihood fitting of a quantitative MRI signal model that is not known analytically, but that is approximated numerically given examples of signals and tissue parameters. Third-party dependencies:
nibabel, numpy, scipy. Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). Email: <fgrussu@vhio.net>.

positional arguments:
  s_file                path of a 4D NIFTI file storing M quantitative MRI measurements acquired (e.g., diffusion MRI at different b-values and/or diffusion time; relaxometry at increasing echo times; etc). Each voxel should
                        contain a signal defined in [0; 1]. In diffusion MRI, this can be obtained by dividing each voxel for the signal at b = 0
  sig_dict              path of a NumPy binary file (.npy) storing a dictionary of synthetic MRI measurements to be used formodel fitting. The file should contain a variable storing a 2D numpy matrix of size Nmicro x M,where
                        Nmicro is the number of example microstructures in the dictionary and Nmeas is the number of MRIsignal measurements. This implies that different example microstructures are arranged along rows,while
                        the different MRI measurements coming from a given microstructure are arranged along columns. NOTE 1: the quantitative MRI protocol used to generate "sdictfile" MUST match that used to acquirethe scan
                        to fit stored in the "mrifile" NIFTI. NOTE 2: the signal dictionary should be defined in [0; 1].
  par_dict              path of a NumPy binary file (.npy) storing a dictionary of tissue parameters corresponding to the signals stored in the signal dictionary "sdictfile", to be used for model fitting. The file should
                        contain a variable storing a 2D numpy matrix of size Nmicro x P, where Nmicro is the number of example microstructures in the dictionary and P is the number of tissue parameters. This implies that
                        different example microstructures are arranged along rows, while the values of the different tissue parameters of each microstructure are arranged along columns.
  out                   root file name of output files. Output NIFTIs will be stored as double-precision floating point images (FLOAT64), and will store the estimated parametric maps. The number of parametric maps outputted
                        depends on number of parameters contained in the tissue parameter dictionary "pdictfile". If that dictionary contains P tissue parameters, then there will be P output parameteric maps (one per each
                        tissue parameter of "pdictfile", in the same order). These will be stored as *_par1.nii, *_par2.nii, ..., *_parP.nii Additionally, an output exit code map will also be stored as *_exit.nii (voxels
                        containing -1: warning, failure in non-linear fitting; voxels containing 0: background; voxels containing 1: successful parameter estimation). If a noise map was provided with the noisefile input
                        parameter, additional output NIFTI files storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information
                        Criterion).

options:
  -h, --help            show this help message and exit
  --mask <file>         3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
  --noise <file>        3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the an estimate of the noise floor when comparing the objective function (offset-Gaussian model).
  --savg <num>          number of signal averages used for MRI data acquisition (default: 1). This parameter is used for the estimation of the noise floor (it is ignored if the option --noise is not used). Note that in some
                        vendors, this parameter is also referred to as number of excitations (NEX).
  --nw <num>            number of values to test for each unknown tissue parameter in the grid search (it must be an integer). If set to 0, the tissue parameter dictionary contained in "par_dict" used to learn the foward
                        model will be used also for the grid search. If > 0, then a uniformly-sampled grid will be generated. Default: 0 (i.e., use the same dictionary to learn the forward model and for the grid search)
  --pmin <list>         comma-separated list of P elements storing the lower bounds for tissue parameters. The length of the list must match the number of tissue parameters contained in the tissue parameter dictionary
                        "par_dict".This option can be used i) to select a subset of the dictionary contained in "par_dict", i.e., to reduce the ranges for estimation for each tissue parameters, or ii) to extend the range of
                        estimation beyond the min/max values contained in the dictionary. If not set, then the lower bounds contained in the dictionary "par_dict" will be used.
  --pmax <list>         comma-separated list of P elements storing the upper bounds for tissue parameters. The length of the list must match the number of tissue parameters contained in the tissue parameter dictionary
                        "par_dict".This option can be used i) to select a subset of the dictionary contained in "par_dict", i.e., to reduce the ranges for estimation for each tissue parameters, or ii) to extend the range of
                        estimation beyond the min/max values contained in the dictionary. If not set, then the upper bounds contained in the dictionary "par_dict" will be used.
  --reg <Lnorm,weight>  comma-separated list of parameters for fitting regularisation specifying i) the type of L-norm (1 for LASSO, 2 for Tikhonov), ii) the weight of the regulariser, ranging in [0.0,1.0]. Default: 2,0.001
                        (L2 norm, with a weight of 0.001). Set 2,0.0 for a standard non-linear least square fitting with no regularisation.
  --ncpu <num>          number of threads to be used for computation (default: 1, single thread)
  --sldim <num>         image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)
  --nlalgo <string>     algorithm to be used for constrained objective function minimisation in non-linear fitting (relevant if --nlfit 1). Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-
                        constr" (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
  --rgalgo <string>     Algorithm to be used to derive a continuous representation of the forward model signal = f(tissue parameters), given the input MRI sequence. Available options: "rbf" (radial basis function regressor
                        based on thin plate splines); "linearND" (piece-wise linear interpolation in N dimensions). Default: "rbf".
  --rgpars <list>       list of comma-separated hyperparameters of the regressor used to derive a continuous signal representation for the forward signal model signal = f(tissue parameters) for the given MRI sequence. A
                        different number of hyperparameters may be needed for each regressor type, as detailed here: for --regalgo "rbf", hyperparameters are the smoothing factor and the degree of the added polynomial
                        (default: --regpars 1.0,1); for --regalgo "linearND", there is one hyperparemter that can be 0 or 1, indicating whethere the regressor should be defined on normalised inputs (if 1) or not (if 0)
                        (default: --regpars 0; anything different from 0 will be treated as 1).
```   
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module mri2micro_dictml:

NAME
    mri2micro_dictml

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

FUNCTIONS
    run(mrifile, sdictfile, pdictfile, output, maskfile=None, noisefile=None, navg=1, Nword=0, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, nlinalgo='trust-constr', rgalgo='rbf', rgpar=None)
        This tool performs maximum-likelihood fitting of a quantitative MRI signal model that is not known analytically, 
        but that is approximated numerically given examples of signals and tissue parameters. 
        
        Third-party dependencies: nibabel, numpy, scipy. 
        
        Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3. 
        
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). 
        Email: <fgrussu@vhio.net>
        
        USAGE
        run(mrifile, sdictfile, pdictfile, output, maskfile=None, noisefile=None, navg=1, ...
            pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, ...
            nlinalgo='trust-constr', rgalgo='rbf', rgpar=None)
        
        * mrifile:    path of a 4D NIFTI file storing M quantitative MRI measurements acquired
                      (e.g., diffusion MRI at different b-values and/or diffusion time; relaxometry at
                      increasing echo times; etc). Each voxel should contain a signal defined in [0; 1].
                                      In diffusion MRI, this can be obtained by dividing each voxel for the signal at b = 0
                      
        * sdictfile:  path of a NumPy binary file (.npy) storing a dictionary of synthetic MRI measurements to be used for
                      model fitting. The file should contain a variable storing a 2D numpy matrix of size Nmicro x M,
                      where Nmicro is the number of example microstructures in the dictionary and Nmeas is the number of MRI 
                      signal measurements.  This implies that different example microstructures are arranged along rows, 
                      while the different MRI measurements coming from a given microstructure are arranged along columns. 
                      NOTE 1: the quantitative MRI protocol used to generate "sdictfile" MUST match that used to acquire 
                      the scan to fit stored in the "mrifile" NIFTI
                      NOTE 2: the signal dictionary should be defined in [0; 1]
        
        * pdictfile:  path of a NumPy binary file (.npy) storing a dictionary of tissue parameters corresponding to the 
                      signals stored in the signal dictionary "sdictfile", to be used for model fitting. 
                      The file should contain a variable storing a 2D numpy matrix of size Nmicro x P,
                      where Nmicro is the number of example microstructures in the dictionary and P is the number of 
                      tissue parameters. This implies that different example microstructures are arranged along rows, while
                      the values of the different tissue parameters of each microstructure are arranged along columns. 
                  
        * output:     root file name of output files. Output NIFTIs will be stored as double-precision floating point images 
                      (FLOAT64), and will store the estimated parametric maps. The number of parametric maps outputted depends
                      on number of parameters contained in the tissue parameter dictionary "pdictfile". If that dictionary
                      contains P tissue parameters, then there will be P output parameteric maps (one per each tissue parameter 
                      of "pdictfile", in the same order). These will be stored as *_par1.nii, *_par2.nii, ...,  *_parP.nii.
                                      
                      
                      Additionally, an output exit code map will also be stored as *_exit.nii (voxels containing -1: warning, 
                      failure in non-linear fitting; voxels containing 0: background; voxels containing 1: 
                      successful parameter estimation). 
                      
                      If a noise map was provided with the noisefile input parameter, additional output NIFTI files 
                      storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), 
                      *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion) 
                      
        * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
                      
        * noisefile:  3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the
                      an estimate of the noise floor when comparing the objective function (offset-Gaussian model).
                      
        * navg:       number of signal averages used for acquisition, which is used to get a more accurate estimate of the
                      noise floor given the estimate of the noise standard deviation (default: 1; ignored if noisefile = None)
                      Note that in some vendors, this parameter is referred to as number of excitations (NEX)
        
        * Nword:      number of values to test for each tissue parameter in the grid search. If set to 0, the
                      tissue parameter dictionary contained in "pdictfile" used to learn the foward model
                      will also be used for the grid search. If Nword > 0, then a uniformly-sampled grid will be generated.
                      Default: 0 (i.e., use the same dictionary to learn the forward model and for the grid search)                   
        
        * pmin:       list or array of P elements storing the lower bounds for tissue parameters. The length of pmin must
                      match the number of tissue parameters contained in the tissue parameter dictionary "pdictfile".
                      This parameter can be used i) to select a subset of the dictionary contained in "pdictfile", i.e., to
                      reduce the ranges of estimation for each tissue parameter, or ii) to extend the range of 
                      estimation beyond min/max values contained in the dictionary. If pmin is not set, then the lower bounds
                      of estimation will be the same as the min values contained in "pdictfile". 
        
        * pmax:       list or array of P elements storing the upper bounds for tissue parameters. The length of pmax must
                      match the number of tissue parameters contained in the tissue parameter dictionary "pdictfile".
                      This parameter can be used i) to select a subset of the dictionary contained in "pdictfile", i.e., to
                      reduce the ranges of estimation for each tissue parameter, or ii) to extend the range of 
                      estimation beyond min/max values contained in the dictionary. If pmax is not set, then the upper bounds
                      of estimation will be the same as the min values contained in "pdictfile". 
        
        * regnorm:    type of L-norm for regularisation (1 for LASSO, 2 for Tikhonov; default: 2) 
            
        * regw:       weight of the regulariser (ranging in [0.0,1.0]; default: 0.001; set it to 0.0 for 
                      standard non-linear least square fitting with no regularisation)      
        
        * nthread:    number of threads to be used for computation (default: 1, single thread)
        
        * slicedim:   image dimension along which parallel computation will be exectued when nthread > 1
                      (can be 0, 1, 2 -- default slicedim=2, implying parallel processing along 3rd 
                      image dimension) 
        
        * nlinalgo:   algorithm to be used for constrained objective function minimisation in non-linear fitting 
                      (relevant if nlinfit = True). Choose among: 
                      "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr". 
                      Default: "trust-constr" - see documentation of scipy.optimize.minimize for 
                      information on each optimisation method)
        
        * rgalgo:     string specifying the algorithm to be used to derive a continuous representation of the forward model 
                      signal = f(tissue parameters), given the input MRI sequence. Available options: 
                      "rbf" (radial basis function regressor based on thin plate splines); 
                      "linearND" (piece-wise linear interpolation in N dimensions). 
                      Default: "rbf".
        
        * rgpar:      list or arry of hyperparameters of the regressor used to derive a continuous signal representation 
                      for the forward signal model signal = f(tissue parameters) for the given MRI sequence. 
                      A different number of hyperparameters may be needed for each regressor type, as detailed here: 
                      rgalgo = "rbf" -->       hyperparameters are the smoothing factor and the degree of the added polynomial.
                                               Default: rgpar = [1.0,1] 
                      regalgo = "linearND" --> there is one hyperparemter that can be 0 or 1 (or False/True), 
                                               indicating whethere the regressor should be defined on normalised inputs 
                                               (if 1 or True) or not (if 0 or False).
                                               Default: rgpar = False (anything different from 0 or False will be treated 
                                               as 1 or True).
        
        Third-party dependencies: nibabel, numpy, scipy.
        
        Developed and validated with versions: 
        - nibabel 3.2.1
        - numpy 1.21.5
        - scipy 1.7.3
        
        Author: Francesco Grussu, Vall d'Hebron Institute of Oncology, November 2022
                <fgrussu@vhio.net>

```
