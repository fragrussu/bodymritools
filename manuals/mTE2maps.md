This script can be called directly from the command line. Its input parameters are the following: 
```
   usage: mTE2maps.py [-h] [--mask <file>] [--noise <file>] [--savg <num>] [--nw <num>] [--pmin <list>] [--pmax <list>] [--reg <Lnorm,weight>] [--ncpu <num>] [--sldim <num>] [--nlalgo <string>] [--modstr <string>]
                   s_file te_file out

This tool estimates the parameters of a magnitude signal decay model, which is fitted to multi-echo gradient or spin echo measurements via regularised maximum-likelihood non-linear least square fitting. Third-party
dependencies: nibabel, numpy, scipy. Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). Email: <fgrussu@vhio.net>.

positional arguments:
  s_file                path of a 4D NIFTI file storing M multi-echo gradient or spin echo MRI measurements acquired at varying TE
  te_file               path of a text file storing a space-separated array of echo times TE in ms, arranged along one row.
  out                   root file name of output files; output NIFTIs will be stored as double-precision floating point images(FLOAT64), and the file names will end in *_S0.nii (signal level at TE = 0 -- S0 = S(TE=0)),
                        *_TAms.nii (relaxation time (T2 or T2*) of water pool A, im ms -- TrexA), *_fA.nii (total signal fraction of water pool A -- fA), *_TBms.nii (relaxation time (T2 or T2*) of water pool B, im ms --
                        TrexB), *_dOmegaBradms-1.nii (off-resonance angular frequency shift of water pool B with respect to pool A, in rad/ms -- dwB), *_fBrel.nii (relative signal fraction of water pool B -- fBrelative; the
                        total signal fraction if fB = fBr x (1 - fA)),*_TBms.nii (relaxation time (T2 or T2*) of water pool B, im ms -- TrexB), *_dOmegaBradms-1.nii (off-resonance angular frequency shift of water pool B with
                        respect to pool A, in rad/ms -- dwB),*_TCms.nii (relaxation time (T2 or T2*) of water pool C, im ms -- TrexC), *_dOmegaCradms-1.nii (off-resonance angular frequency shift of water pool C with respect
                        to pool A, in rad/ms -- dwC), *_exit.nii (voxel-wise exit code; -1: warning, failure in non-linear fitting; 0: background; 1 successful parameter estimation). If a noise map was provided with the
                        noisefile input parameter, additional output NIFTI filesstoring quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike
                        Information Criterion). The number of parametric maps finally stored depends on the model specified with input parameter modstr(see below). These will be: TrexA, S0 for model "MonoExp" (mono-
                        exponential decay); TrexA, fA, TrexB, S0 for model "BiExp" (bi-exponential decay); TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres" (bi-exponential decay with off-resonance effects); TrexA, fA,
                        TrexB, fBrelative, TrexC, S0 for model "TriExp" (tri-exponential decay); TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres" (tri-exponential decay with off-resonance effects)

options:
  -h, --help            show this help message and exit
  --mask <file>         3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
  --noise <file>        3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the expected Rician noise floor.
  --savg <num>          number of signal averages used for MRI data acquisition (default: 1). This parameter is used for the estimation of the noise floor (it is ignored if the option --noise is not used). Note that in some
                        vendors, this parameter is also referred to as number of excitations (NEX).
  --nw <num>            number of values to test for each unknown tissue parameter in the grid search (it must be an integer; default 3)
  --pmin <list>         list storing the lower bounds for tissue parameters. These are: TrexA, S0 for model "MonoExp"; TrexA, fA, TrexB, S0 for model "BiExp"; TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres"; TrexA, fA,
                        TrexB, fBrelative, TrexC, S0 for model "TriExp"; TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres".The symbols stand for: -- TrexA -> relaxation time of water pool A (ms) --
                        fA -> signal fraction ofwater pool A -- TrexB -> relaxation time of water pool B (ms) -- fBrelative -> relative signal fraction of water pool B -- TrexC -> relaxation time of water pool C (ms) -- dwB
                        -> off-resonance angular frequency shift of water pool B with respect to A (rad/ms) -- dwC -> off-resonance angular frequency shift of water pool C with respect to A (rad/ms) -- S0 --> signal at TE =
                        0, with respect to the maximum of the measured signal(parameterisation: S(TE = 0) = S0*max(measured_signal)). Default:"2.0,0.8" for parameters TrexA, S0 in model "MonoExp"; "2.0,0.01,35.0,0.8" for
                        parameters TrexA, fA, TrexB, S0 in model "BiExp"; "2.0,0.01,35.0,-0.4,0.8" for parameters TrexA, fA, TrexB, dwB, S0 in model "BiExpOffres"; "2.0,0.01,15.0,0.01,35.0,0.8" for parameters TrexA, fA,
                        TrexB, fBrelative, TrexC, S0 in model "TriExp"; "2.0,0.01,15.0,0.01,-0.4,35.0,-0.4,0.8" for parameters TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 inmodel "TriExpOffres". For reference, note
                        that the water-fat frequency shit at 3T is equal to 2.8 rad/ms, and here smaller frequency shifts are to be expected. For more information on the models, please look at option modstr below.
  --pmax <list>         list or array storing the upper bounds for tissue parameters. These are: TrexA, S0 for model "MonoExp"; TrexA, fA, TrexB, S0 for model "BiExp"; TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres";
                        TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp"; TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres". The symbols stand for: -- TrexA -> relaxation time of water pool
                        A (ms) -- fA -> signal fraction ofwater pool A -- TrexB -> relaxation time of water pool B (ms) -- fBrelative -> relative signal fraction of water pool B -- TrexC -> relaxation time of water pool C
                        (ms) -- dwB -> off-resonance angular frequency shift of water pool B with respect to A (rad/ms) -- dwC -> off-resonance angular frequency shift of water pool C with respect to A (rad/ms) -- S0 -->
                        signal at TE = 0, with respect to the maximum of the measured signal(parameterisation: S(TE = 0) = S0*max(measured_signal)). Default:"150.0,50.0" for parameters TrexA, S0 in model "MonoExp";
                        "35.0,0.99,150.0,50.0" for parameters TrexA, fA, TrexB, S0 in model "BiExp"; "35.0,0.99,150.0,0.4,50.0" for parameters TrexA, fA, TrexB, dwB, S0 in model "BiExpOffres";
                        "15.0,0.99,35.0,0.99,150.0,50.0" for parameters TrexA, fA, TrexB, fBrelative, TrexC, S0 in model "TriExp"; "15.0,0.99,35.0,0.99,0.4,150.0,0.4,50.0" for parameters TrexA, fA, TrexB, fBrelative, dwB,
                        TrexC, dwC, S0 inmodel "TriExpOffres". For reference, note that the water-fat angular frequency shit at 3T is equal to 2.8 rad/ms, and here smaller frequency shifts are to be expected. For more
                        information on the models, please look at option modstr below.
  --reg <Lnorm,weight>  comma-separated list of parameters for fitting regularisation specifying i) the type of L-norm (1 for LASSO, 2 for Tikhonov), ii) the weight of the regulariser, ranging in [0.0,1.0]. Default: 2,0.001
                        (L2 norm, with a weight of 0.001). Set 2,0.0 for a standard non-linear least square fitting with no regularisation.
  --ncpu <num>          number of threads to be used for computation (default: 1, single thread)
  --sldim <num>         image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)
  --nlalgo <string>     algorithm to be used for constrained objective function minimisation in non-linear fitting (relevant if --nlfit 1). Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-
                        constr" (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
  --modstr <string>     string specifying the signal model to fit. Choose among "MonoExp", "BiExp", "BiExpOffres": - "MonoExp" (mono-exponential T2 or T2* decay), - "BiExp" (bi-exponential T2 or T2* decay; for T2*, the
                        components are considered to be perfectly in phase), - "BiExpOffres" (bi-exponential T2* decay, with one component being off-resonance compared to the other; meaningful only for multi-echo gradient
                        echo measurements -- DO NOT USE ON SPIN ECHOES); - "TriExp" (tri-exponential T2 or T2* decay; for T2*, the components are considered to be perfectly in phase), - "TriExpOffres" (tri-exponential T2*
                        decay, with components B and C being off-resonance compared to reference component A; meaningful only for multi-echo gradient echo measurements -- DO NOT USE ON SPIN ECHOES); Default: "MonoExp". Note:
                        models "BiExpOffres" and "TriExpOffres" are practical implementations of Susceptibility Perturbation MRI by Santiago I et al, Cancer Research (2019) 79 (9): 2435-2444, doi:
                        10.1158/0008-5472.CAN-18-3682 (paper link: https://doi.org/10.1158/0008-5472.CAN-18-3682).
```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module mTE2maps:

NAME
    mTE2maps

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

FUNCTIONS
    run(mrifile, tefile, output, maskfile=None, noisefile=None, navg=1, Nword=3, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, nlinalgo='trust-constr', modstr='MonoExp')
        This tool estimates the parameters of a magnitude signal decay model, which is fitted to 
        multi-echo gradient or spin echo measurements via regularised maximum-likelihood 
        non-linear least square fitting
        
        Third-party dependencies: nibabel, numpy, scipy. 
        Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3.
        
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). 
        Email: <fgrussu@vhio.net>.
        
        USAGE
        run(mrifile, tefile, output, maskfile=None, noisefile=None, s0file=None, navg=1, ... 
            Nword=6, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, ...
            nlinalgo='trust-constr', modstr='MonoExp')
        
        * mrifile:    path of a 4D NIFTI file storing M multi-echo gradient or spin echo MRI measurements acquired at varying TE
                      
        * tefile:     path of a text file storing a space-separated array of echo times TE in ms, arranged along one row 
                      
        * output:     root file name of output files; output NIFTIs will be stored as double-precision floating point images 
                      (FLOAT64), and the file names will end in 
                      *_S0.nii (signal level at TE = 0 -- S0 = S(TE=0)),
                      *_TAms.nii (relaxation time (T2 or T2*) of water pool A, im ms -- TrexA), 
                      *_fA.nii (total signal fraction of water pool A -- fA),
                      *_TBms.nii (relaxation time (T2 or T2*) of water pool B, im ms -- TrexB),
                      *_dOmegaBradms-1.nii (off-resonance angular frequency shift of water pool B with respect to pool A, in rad/ms -- dwB), 
                      *_fBrel.nii (relative signal fraction of water pool B -- fBrelative; the total signal fraction if fB = fBr x (1 - fA)),
                      *_TBms.nii (relaxation time (T2 or T2*) of water pool B, im ms -- TrexB),
                      *_dOmegaBradms-1.nii (off-resonance angular frequency shift of water pool B with respect to pool A, in rad/ms -- dwB),
                      *_TCms.nii (relaxation time (T2 or T2*) of water pool C, im ms -- TrexC),
                      *_dOmegaCradms-1.nii (off-resonance angular frequency shift of water pool C with respect to pool A, in rad/ms -- dwC), 
                      *_exit.nii (voxel-wise exit code; -1: warning, failure in non-linear fitting; 0: background; 
                      1 successful parameter estimation). 
                      
                      If a noise map was provided with the noisefile input parameter, additional output NIFTI files 
                      storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), 
                      *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion). 
                      
                      The number of parametric maps finally stored depends on the model specified with input parameter modstr
                      (see below). These will be:
                      TrexA, S0 for model "MonoExp" (mono-exponential decay); 
                      TrexA, fA, TrexB, S0 for model "BiExp" (bi-exponential decay); 
                      TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres" (bi-exponential decay with off-resonance effects)
                      TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp" (tri-exponential decay); 
                      TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres" (tri-exponential decay with off-resonance effects)
        
        * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
                      
        * noisefile:  3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the
                      expected Rician noise floor.  
                                                              
        * navg:       number of signal averages used for acquisition, which is used to get a more accurate estimate of the
                      noise floor given the estimate of the noise standard deviation (default: 1; ignored if noisefile = None)
                      Note that in some vendors, this parameter is referred to as number of excitations (NEX)
                      
        * Nword:      number of values to test for each tissue parameter in the grid search (default: 3)
        
        * pmin:       list or array storing the lower bounds for tissue parameters. These are: 
                      TrexA, S0 for model "MonoExp"; 
                      TrexA, fA, TrexB, S0 for model "BiExp"; 
                      TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres";
                      TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp"; 
                      TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres". 
                      
                      The symbols stand for: 
                      -- TrexA -> relaxation time of water pool A (ms)
                      -- fA -> signal fraction ofwater pool A
                      -- TrexB -> relaxation time of water pool B (ms)
                      -- fBrelative -> relative signal fraction of water pool B
                      -- TrexC -> relaxation time of water pool C (ms)
                      -- dwB -> off-resonance angular frequency shift of water pool B with respect to A (rad/ms)
                      -- dwC -> off-resonance angular frequency shift of water pool C with respect to A (rad/ms)
                      -- S0 --> signal at TE = 0, with respect to the maximum of the measured signal
                      (parameterisation: S(TE = 0) = S0*max(measured_signal))
        
                      Default: 
                      "2.0,0.8" for parameters TrexA, S0 in model "MonoExp"; 
                      "2.0,0.01,35.0,0.8" for parameters TrexA, fA, TrexB, S0 in model "BiExp"; 
                      "2.0,0.01,35.0,-0.4,0.8" for parameters TrexA, fA, TrexB, dwB, S0 in model "BiExpOffres";
                      "2.0,0.01,15.0,0.01,35.0,0.8" for parameters TrexA, fA, TrexB, fBrelative, TrexC, S0 in model "TriExp"; 
                      "2.0,0.01,15.0,0.01,-0.4,35.0,-0.4,0.8" for parameters TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 in 
                      model "TriExpOffres"
        
                      For reference, note that the water-fat frequency shit at 3T is equal to 2.8 rad/ms.
                      Here smaller frequency shifts are to be expected.
                      
                      For more information on the models, please look at input parameter modstr below.
        
        * pmax:       list or array storing the upper bounds for tissue parameters. These are: 
                      TrexA, S0 for model "MonoExp"; 
                      TrexA, fA, TrexB, S0 for model "BiExp"; 
                      TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres";
                      TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp"; 
                      TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres". 
                      
                      The symbols stand for: 
                      -- TrexA -> relaxation time of water pool A (ms)
                      -- fA -> signal fraction ofwater pool A
                      -- TrexB -> relaxation time of water pool B (ms)
                      -- fBrelative -> relative signal fraction of water pool B
                      -- TrexC -> relaxation time of water pool C (ms)
                      -- dwB -> off-resonance angular frequency shift of water pool B with respect to A (rad/ms)
                      -- dwC -> off-resonance angular frequency shift of water pool C with respect to A (rad/ms)
                      -- S0 --> signal at TE = 0, with respect to the maximum of the measured signal
                      (parameterisation: S(TE = 0) = S0*max(measured_signal))
        
                      Default:
                      "150.0,50.0" for parameters TrexA, S0 in model "MonoExp"; 
                      "35.0,0.99,150.0,50.0" for parameters TrexA, fA, TrexB, S0 in model "BiExp"; 
                      "35.0,0.99,150.0,0.4,50.0" for parameters TrexA, fA, TrexB, dwB, S0 in model "BiExpOffres";
                      "15.0,0.99,35.0,0.99,150.0,50.0" for parameters TrexA, fA, TrexB, fBrelative, TrexC, S0 in model "TriExp"; 
                      "15.0,0.99,35.0,0.99,0.4,150.0,0.4,50.0" for parameters TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 in 
                      model "TriExpOffres"
               
                      For reference, note that the water-fat frequency shit at 3T is equal to 2.8 rad/ms.
                      Here smaller frequency shifts are to be expected.
        
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
        
        * modstr:     string specifying the signal model to fit. Choose among "MonoExp", "BiExp", "BiExpOffres": 
                      - "MonoExp" (mono-exponential T2 or T2* decay), 
                      - "BiExp" (bi-exponential T2 or T2* decay; for T2*, the components are considered to be perfectly in phase), 
                      - "BiExpOffres" (bi-exponential T2* decay, with one component being off-resonance compared to the other; meaningful only for 
                         multi-echo gradient echo measurements -- DO NOT USE ON SPIN ECHOES);
                      - "TriExp" (tri-exponential T2 or T2* decay; for T2*, the components are considered to be perfectly in phase), 
                      - "TriExpOffres" (tri-exponential T2* decay, with components B and C being off-resonance compared to reference component A; 
                                         meaningful only for multi-echo gradient echo measurements -- DO NOT USE ON SPIN ECHOES);
                      
                      Default: "MonoExp". 
                      
                      Note: models "BiExpOffres" and "TriExpOffres" are practical implementations of Susceptibility Perturbation MRI by 
                      Santiago I et al, Cancer Research (2019) 79 (9): 2435-2444, doi: 10.1158/0008-5472.CAN-18-3682 
                      https://doi.org/10.1158/0008-5472.CAN-18-3682
        
                
        Third-party dependencies: nibabel, numpy, scipy.
        
        Developed and validated with versions: 
        - nibabel 3.2.1
        - numpy 1.21.5
        - scipy 1.7.3
        
        Author: Francesco Grussu, Vall d'Hebron Institute of Oncology, February 2024
                <fgrussu@vhio.net>
```
