This script can be called directly from the command line. Its input parameters are the following: 
```
usage: pgse2sphereinex.py [-h] [--mask <file>] [--noise <file>] [--s0ref <file>] [--savg <num>] [--nw <num>] [--pmin <list>] [--pmax <list>] [--reg <Lnorm,weight>] [--ncpu <num>] [--sldim <num>] [--nlalgo <string>]
                          [--modstr <string>]
                          s_file scheme_file out

This tool estimates the parameters of a 2-compartment model of restricted diffusion within spheres and extra-cellular hindered diffusion, via regularised non-linear optimisation of a likelihood function under an offset-
Gaussian noise model.Third-party dependencies: nibabel, numpy, scipy. Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO).
Email: <fgrussu@vhio.net>.

positional arguments:
  s_file                path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values and diffusion times
  scheme_file           path of a text file storing information on b-values and diffusion times corresponding to each volume of s_file. The acquisition must be a standard pulsed gradient spin echo (PGSE, also known as single
                        linear diffusion encoding). This file must contain a space-separated array of 3 x M elements, arranged along 3 lines (first line: b-values in s/mm2; second line: gradient duration delta in ms; third
                        line: gradient separation Delta in ms).
  out                   root file name of output files; output NIFTIs will be stored as double-precision floating point images (FLOAT64), and the file names will end in *_Lum.nii (cell diameter L in um), *_D0um2ms-1.nii
                        (intrinsic intra-cellular diffusivity D0 in um2/ms), *_Dexinfum2ms-1.nii (extra-cellular apparent diffusion coefficient parameter Dexinf in um2/ms), *_Betaum2.nii (extra-cellular apparent diffusion
                        coefficient parameter Beta -- note that the extra-cellular apparent diffusion coefficient Dex is written as Dex = Dexinf + Beta/t, where t is the gradient separation of measurements kept after b-value
                        thresholding (see input parameter bth), *_fin.nii (intra-cellular tissue signal fraction Fin), *_S0.nii (non-DW signal level S0), *_cellsmm-2.nii (2D histology-like cellularity C in cells/mm2),
                        *_cellsmm-3.nii (cellularity C in cells/mm3), *_exit.nii (voxel-wise exit code; -1: warning, failure in non-linear fitting; 0: background; 1 successful parameter estimation). If a noise map was
                        provided with the noisefile input parameter, additional output NIFTI files storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), *_BIC.nii (Bayesian Information Criterion), and
                        *_AIC.nii (Akaike Information Criterion). The number of parametric maps outputted depends on the model specified with input parameter modstr (see below). These will be: L, D0, Fin, S0 for model "Din";
                        L, D0, Dexinf, Fin, S0 for model "DinDex"; L, D0, Dexinf, Beta, Fin, S0 for model "DinDexTD"

options:
  -h, --help            show this help message and exit
  --mask <file>         3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
  --noise <file>        3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the expected Rician noise floor.
  --s0ref <file>        3D reference non-DW signal S0 in NIFTI format. Relevant only if the fitted model is "Din" (see modstr below). If provided, S0_Din, i.e., the non-DW signal level estimated via "Din" fitting, will be
                        compared to S0 to estimate the intra-cellular signal fraction F as F = S0_Din/S0. F will be stored as a NIFTI map as for models "DinDex" and "DinDexTD"
  --savg <num>          number of signal averages used for MRI data acquisition (default: 1). This parameter is used for the estimation of the noise floor (it is ignored if the option --noise is not used). Note that in some
                        vendors, this parameter is also referred to as number of excitations (NEX).
  --nw <num>            number of values to test for each unknown tissue parameter in the grid search (it must be an integer; default 10)
  --pmin <list>         list or array storing the lower bounds for tissue parameters. These are: L,D0,S0 for model "Din"; L,D0,F,Dexinf,S0 for model "DinDex"; L,D0,F,Dexinf,Beta,S0 for model "DinDexTD". The symbols stand
                        for: L -> cell size (diameter), in (um); D0 -> intrinsic intra-cell diffusivity in (um2/ms); F -> intra-cellular signal fraction; Dexinf -> long-time extra-cellular apparent diffusion coefficient, in
                        (um2/ms); Beta -> extra-cellular diffusion-time dependence coefficient in (um2); S0 -> non-DW signal level, with respect to the maximum measured signal (parametrisation: S(b = 0) = S0*max(S)).. Note
                        that the extra-cellular apparent diffusion coefficient is written as Dex = Dexinf + Beta/t, where t is the gradient separation. Default: "8.0,0.8,0.01" for model "Din"; "8.0,0.8,0.01,0.01,0.6" for
                        model "DinDex"; "8.0,0.8,0.01,0.01,0.01,0.6" for model "DinDexTD". For more information on the models, please look at input parameter modstr below.
  --pmax <list>         list or array storing the lower bounds for tissue parameters. These are: L,D0,S0 for model "Din"; L,D0,F,Dexinf,S0 for model "DinDex"; L,D0,F,Dexinf,Beta,S0 for model "DinDexTD". The symbols stand
                        for: L -> cell size (diameter), in (um); D0 -> intrinsic intra-cell diffusivity in (um2/ms); F -> intra-cellular signal fraction; Dexinf -> long-time extra-cellular apparent diffusion coefficient, in
                        (um2/ms); Beta -> extra-cellular diffusion-time dependence coefficient in (um2); S0 -> non-DW signal level, with respect to the maximum measured signal (parametrisation: S(b = 0) = S0*max(S)).. Note
                        that the extra-cellular apparent diffusion coefficient is written as Dex = Dexinf + Beta/t, where t is the gradient separation. Default: "40.0,3.0,1.4" for model "Din"; "40.0,3.0,1.0,3.0,1.4" for
                        model "DinDex"; "40.0,3.0,1.0,3.0,10.0,1.4" for model "DinDexTD". For more information on the models, please look at input parameter modstr below.
  --reg <Lnorm,weight>  comma-separated list of parameters for fitting regularisation specifying i) the type of L-norm (1 for LASSO, 2 for Tikhonov), ii) the weight of the regulariser, ranging in [0.0,1.0]. Default: 2,0.001
                        (L2 norm, with a weight of 0.001). Set 2,0.0 for a standard non-linear least square fitting with no regularisation.
  --ncpu <num>          number of threads to be used for computation (default: 1, single thread)
  --sldim <num>         image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)
  --nlalgo <string>     algorithm to be used for constrained objective function minimisation in non-linear fitting (relevant if --nlfit 1). Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-
                        constr" (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
  --modstr <string>     string specifying the signal model to fit. Choose among: Din (extra-vascular signal dominated by intra-cellular diffusion), DinDex (extra-vascular signal features both intra-cellular and extra-
                        cellular contributions, without diffusion time dependence in the extra-cellular ADC), DinDexTD (extra-vascular signal features both intra-cellular and extra-cellular contributions, with diffusion time
                        dependence in the extra-cellular ADC). Default: DinDex. Intra-cellular diffusion is modelled using the Gaussian Phase Approximation (GPA) formula for diffusion within sphere of Balinov et al. "The NMR
                        self-diffusion method applied to restricted diffusion. Simulation of echo attenuation from molecules in spheres and between planes." Journal of Magnetic Resonance, Series A 104.1 (1993): 17-25, doi:
                        10.1006/jmra.1993.1184
```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module pgse2sphereinex:

NAME
    pgse2sphereinex

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

FUNCTIONS
    run(mrifile, mriseq, output, maskfile=None, noisefile=None, s0file=None, navg=1, Nword=6, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, nlinalgo='trust-constr', modstr='DinDex')
        This tool estimates the parameters of a 2-compartment model of restricted diffusion within 
        spheres and extra-cellular hindered diffusion, via regularised non-linear optimisation of 
        a likelihood function under an offset-Gaussian noise model
        
        Third-party dependencies: nibabel, numpy, scipy. 
        Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3.
        
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). 
        Email: <fgrussu@vhio.net>.
        
        USAGE
        run(mrifile, mriseq, output, maskfile=None, noisefile=None, s0file=None, navg=1, ... 
            Nword=6, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, ...
            nlinalgo='trust-constr', modstr='DinDex')
        
        * mrifile:    path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values 
                      and diffusion times
                      
        * mriseq:     path of a text file storing information on b-values and diffusion times  corresponding to each 
                      volume of s_file. The acquisition must be a standard pulsed gradient spin echo
                      (PGSE, also known as single linear diffusion encoding).
                      This file must contain a space-separated array of 3 x M elements, arranged along 3 lines 
                      (first line: b-values in s/mm2; 
                      second line: gradient duration delta in ms; 
                      third line: gradient separation Delta in ms).
                      
        * output:     root file name of output files; output NIFTIs will be stored as double-precision floating point images 
                      (FLOAT64), and the file names will end in 
                      *_Lum.nii (cell diameter L in um), 
                      *_D0um2ms-1.nii (intrinsic intra-cellular diffusivity D0 in um2/ms), 
                      *_Dexinfum2ms-1.nii (extra-cellular apparent diffusion coefficient parameter Dexinf in um2/ms), 
                      *_Betaum2.nii (extra-cellular apparent diffusion coefficient parameter Beta -- note that the 
                      extra-cellular apparent diffusion coefficient Dex is written as Dex = Dexinf + Beta/t, where t is 
                      the gradient separation, 
                      *_fin.nii (intra-cellular tissue signal fraction F),
                      *_S0.nii (non-DW signal level S0), 
                      *_cellsmm-2.nii (2D histology-like cellularity C in cells/mm2), 
                      *_cellsmm-3.nii (cellularity C in cells/mm3), 
                      *_exit.nii (voxel-wise exit code; -1: warning, failure in non-linear fitting; 0: background; 
                      1 successful parameter estimation). 
                      
                      If a noise map was provided with the noisefile input parameter, additional output NIFTI files 
                      storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), 
                      *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion). 
                      
                      The number of parametric maps outputted depends on the model specified with input parameter modstr
                      (see below). These will be: 
                      L, D0, S0 for model "Din"; 
                      L, D0, Dexinf, F, S0 for model "DinDex"; 
                      L, D0, Dexinf, Beta, F, S0 for model "DinDexTD"
                      
        * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
                      
        * noisefile:  3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the
                      expected Rician noise floor.
        
        * s0file:     3D reference non-DW signal S0 in NIFTI format. Relevant only if the fitted model is "Din" (see modstr below). 
                          If provided, S0_Din, i.e., the non-DW signal level estimated via "Din" fitting, will be compared to 
                                      S0 to estimate the intra-cellular signal fraction F as F = S0_Din/S0. F will be stored as a NIFTI map
                                      as for models "DinDex" and "DinDexTD"                   
                                                              
        * navg:       number of signal averages used for acquisition, which is used to get a more accurate estimate of the
                      noise floor given the estimate of the noise standard deviation (default: 1; ignored if noisefile = None)
                      Note that in some vendors, this parameter is referred to as number of excitations (NEX)
                      
        * Nword:      number of values to test for each tissue parameter in the grid search (default: 6)
        
        * pmin:       list or array storing the lower bounds for tissue parameters. These are: 
                      L,D0,S0 for model "Din"; 
                      L,D0,F,Dexinf,S0 for model "DinDex"; 
                      L,D0,F,Dexinf,Beta,S0 for model "DinDexTD". 
                      
                      The symbols stand for: 
                      -- L -> cell size (cell diameter) (um)
                      -- D0 -> intrinsic intra-cell diffusivity (um2/ms)
                      -- F -> intra-cellular signal fraction
                      -- Dexinf -> long-time extra-cellular apparent diffusion coefficient (um2/ms), 
                      -- Beta -> extra-cellular diffusion-time dependence coefficient (um2) 
                      Note that the extra-cellular apparent diffusion coeffficient is written as 
                      Dex = Dexinf + Beta/t, where t is the diffusion time (~ gradient separation) 
                      -- S0 --> non-DW signal, with respect to the maximum of the measured signal
                      (parameterisation: S(b = 0) = S0*max(measured_signal))
        
                      Default: 
                      "8.0,0.8,0.01" for model "Din"; 
                      "8.0,0.8,0.01,0.01,0.6" for model "DinDex"; 
                      "8.0,0.8,0.01,0.01,0.01,0.6" for model "DinDexTD"
                      
                      For more information on the models, please look at input parameter modstr below.
        
        * pmax:       list or array storing the upper bounds for tissue parameters. These are: 
                      L,D0,F,S0 for model "Din"; 
                      L,D0,F,Dexinf,S0 for model "DinDex"; 
                      L,D0,F,Dexinf,Beta,S0 for model "DinDexTD". 
                      
                      The symbols stand for: 
                      -- L -> cell size (cell diameter) (um)
                      -- D0 -> intrinsic intra-cell diffusivity (um2/ms)
                      -- F -> intra-cellular signal fraction
                      -- Dexinf -> long-time extra-cellular apparent diffusion coefficient (um2/ms), 
                      -- Beta -> extra-cellular diffusion-time dependence coefficient (um2) 
                      Note that the extra-cellular apparent diffusion coeffficient is written as 
                      Dex = Dexinf + Beta/t, where t is the diffusion time (~ gradient separation)
                      -- S0 --> non-DW signal, with respect to the maximum of the measured signal
                      (parameterisation: S(b = 0) = S0*max(measured_signal))
                      
                      Default: 
                      "40.0,3.0,1.4" for model "Din"; 
                      "40.0,3.0,1.0,3.0,1.4" for model "DinDex"; 
                      "40.0,3.0,1.0,3.0,10.0,1.4" for model "DinDexTD"
                      
                      For more information on the models, please look at input parameter modstr below.            
        
        * regnorm:    type of L-norm for regularisation (1 for LASSO, 2 for Tikhonov; default: 2) 
            
        * regw:       weight of the regulariser (ranging in [0.0,1.0]; default: 0.001; set it to 0.0 for 
                      standard non-linear least square fitting with no regularisation)  
        
        * nthread:    number of threads to be used for computation (default: 1, single thread)
        
        * slicedim:   image dimension along which parallel computation will be exectued when nthread > 1
                      (can be 0, 1, 2 -- default slicedim=2, implying parallel processing along 3rd 
                      image dimension)
        
        * nlinalgo:   algorithm to be used for constrained objective function minimisation in non-linear fitting. 
                      Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" 
                      (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
        
        * modstr:     string specifying the signal model to fit. Choose among: 
                      - "Din" (extra-vascular signal dominated by intra-cellular diffusion), 
                      - "DinDex" (extra-vascular signal features both intra-cellular and extra-cellular contributions, without diffusion time 
                      dependence in the extra-cellular ADC), 
                      - "DinDexTD" (extra-vascular signal features both intra-cellular and extra-cellular contributions, with diffusion time 
                      dependence in the extra-cellular ADC).
                      
                      Default: "DinDex". 
                      
                      Intra-cellular diffusion is modelled using the Gaussian Phase Approximation (GPA) formula for diffusion within sphere 
                      of Balinov et al. "The NMR self-diffusion method applied to restricted diffusion. Simulation of echo 
                      attenuation from molecules in spheres and between planes." 
                      Journal of Magnetic Resonance, Series A 104.1 (1993): 17-25, doi: 10.1006/jmra.1993.1184            
                
        Third-party dependencies: nibabel, numpy, scipy.
        
        Developed and validated with versions: 
        - nibabel 3.2.1
        - numpy 1.21.5
        - scipy 1.7.3
        
        Author: Francesco Grussu, Vall d'Hebron Institute of Oncology, November 2022
                <fgrussu@vhio.net>
```   
