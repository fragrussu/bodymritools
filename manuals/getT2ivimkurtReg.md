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
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module getT2ivimkurtReg:

NAME
    getT2ivimkurtReg

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

FUNCTIONS
    FitModel(sig_nifti, seq_text, output_rootname, ncpu=1, mask_nifti=None, sigma_nifti=None, nex=1, pMin=[0.8, 0.0, 4.0, 150.0, 0.2, 0.0, 20.0], pMax=[12.0, 1.0, 100.0, 250.0, 3.0, 5.0, 140.0], gridres=3, wtk=0.0)
        Fit the T2-IVIM-Kurtosis model s(b,TE) = s0*(fv*exp(-bDv -TE/T2v) + (1-fv)*exp(-bDt +(Kt/6)*b^2*Dt^2 -TE/T2t ))
        
        DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel
        
        INTERFACES
        FitModel(sig_nifti,seq_text,output_rootname,ncpu=1,mask_nifti=None,sigma_nifti=None,nex=1,pMin=defmin,pMax=defmax,gridres=4,wtk=0.0)
         
        PARAMETERS
        - sig_nifti: path of a Nifti file storing the multi b-value data as 4D data.
        - seq_text: path of a text file storing the b-values (s/mm^2) used to acquire the data.
        - output_rootname: base name of output files. Output files will end in 
                           "_s0.nii"      --> apparent proton density
                           "_fv.nii"      --> vascular signal fraction
                           "_dv.nii"      --> apparent vascular diffusion coefficient (um2/ms)
                           "_t2v.nii"     --> vascular T2 (ms)
                           "_dt.nii"      --> apparent tissue diffusion coefficient (um2/ms)
                           "_kt.nii"      --> apparent tissue excess kurtosis coefficient
                           "_t2t.nii"     --> tissue T2 (ms)
                           "_fobj.nii"    --> fitting objective function
                           "_exit.nii"    --> fitting exit code (1: success; -1: warning; 0: background)
                           
                        Output files will be stored as double-precision floating point (FLOAT64)
                    
        - ncpu: number of processors to be used for computation (optional; default: 1)
        - mask_nifti: path of a Nifti file storing a binary mask, where 1 flgas voxels where the 
                      signal model needs to be fitted, and 0 otherwise (optional; default: None, i.e. no mask)
        - sigma_nifti: path of a Nifti file storing voxel-wise estimates of the noise standard deviation, which
                       will be used to calculate the noise floor for offset-Gaussian likelihood maximisation
                       (optional; default: None, i.e. standard Gaussian noise)
        - nex: number of signal averages (NSA), also known as number of excitations (NEX) (optional, and relevant 
               only if a noise standard deviation map is provided via sigma_nifti; default: 1). 
               Note that trace-DW imaging has an intrinsic number of signal averages of 3, since images for 
               three mutually orthogonal gradients are acquired and averaged.
        - pMin: lower bound for tissue parameters. Default pMin=defmin, with defmin
                elements being (in order): 
                       S0 = 0.8 for apparent proton density (true S0 bound obtained multiplyig this number of max(signal))
                       fv = 0.0 for vascular signal fraction                           
                       Dv = 4 um2/ms for the vascular apparent diffusion coefficient
                       T2v = 150 ms for the vascular T2                    
                       Dt = 0.2 um2/ms for the tissue apparent diffusion coefficient
                       Kt = 0.0 for the tissue apparent excess kurtosis coefficient
                       T2t = 20.0 for the tissue T2
        - pMax: upper bound for tissue parameters. Default pMax=defmax, with defmax
                elements being (in order):
                       S0 = 12.0 for apparent proton density (true S0 bound obtained multiplyig this number of max(signal))
                       fv = 1.0 for vascular signal fraction                        
                       Dv = 100 um2/ms for the vascular apparent diffusion coefficient
                       T2v = 250 ms for the vascular T2                    
                       Dt = 3.0 um2/ms for the tissue apparent diffusion coefficient
                       Kt = 5.0 for the tissue apparent excess kurtosis coefficient
                       T2t = 140.0 for the tissue T2        
        - gridres: grid depth for grid search (i.e., number of test values for each tissue parameter; 
                   default gridres = 3)
        - wtk: Tikhonov regularisation weight (must be <=0; default 0.0, implying no regularisation. 
               Note that an effective regularisation weight value will depend on the range on which the MRI signal 
               is defined: for instance, for an MRI signal varying in the range [0; SMAX], resonable regularisation weights 
               are of the order of approx SMAX/3)
                    
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain
               <fgrussu@vhio.net>
    
    FitSlice(data)
        Fit T2-IVIM-Kurtosis one MRI slice stored as a 3D numpy array (2 image dimensions; 1 for multiple measurements)  
        
        DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel
        
        INTERFACE
        data_out = FitSlice(data)
         
        PARAMETERS
        - data: a list of 7 elements, such that
                data[0] is a 3D numpy array contaning the data to fit. The first and second dimensions of data[0]
                        are the slice first and second dimensions, whereas the third dimension of data[0] stores
                        different MRI measurements
                data[1] is a 2D numpy matrix, with 1st row listing b-values (in s/mm^2) and 2nd row listing TE (in ms)
                data[2] is a 2D numpy array contaning the fitting mask within the MRI slice (see FitModel())
                data[3] contains the lower bound for the tissue parameters. Please see signal_gen() and FitModel()
                data[4] contains the upper bound for the tissue parameters. Please see signal_gen() and FitModel()
                data[5] is a 2D numpy array contaning the noise standard deviation map within the MRI slice (see FitModel())
                data[6] is the number of signal averages (NEX or NSA; note that trace-DW imaging has an intrinsic NEX = 3)
                data[7] is the number of values to test for each tissue parameter in the grid search (see GridSearch())
                data[8] is the Tikhonov regularisation weight (see FitModel())
                data[9] is a scalar containing the index of the MRI slice in the 3D volume
        
        RETURNS
        - data_out: a list of 8 elements, such that
                       data_out[0] = S0 estimates (apparent proton density)
                       data_out[1] = fv estimates (vascular signal fraction, ranging in 0-1)                            
                       data_out[2] = Dv estimates (apparent diffusion coefficient of vascular water, in s/mm2)
                       data_out[3] = T2v estimates (T2 of vascular water, in ms)                      
                       data_out[4] = Dt estimates (apparent diffusion coefficient of tissue, in s/mm2)
                       data_out[5] = Kt estimates (apparent excess kurtosis coefficient of tissue, dimensionless)
                       data_out[6] = T2t estimates (T2 of tissue, in ms)
                       data_out[7] = value of the minimised objective function
                       data_out[8] = fitting exit code
                       data_out[9] = index of the MRI slice in the 3D volume (same as data[3])
        
                Fitted parameters in data_out will be stored as double-precision floating point (FLOAT64)
         
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain
               <fgrussu@vhio.net>
    
    Fobj(tissue_par, mri_seq, meas, sgmnoise, navg, parmin, parmax, wtikh)
        Fitting objective function to estimate T2-IVIM-Kurtosis tissue parameters              
        
        DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel
            
        INTERFACE
        fobj = Fobj(tissue_par,mri_seq,meas,sgmnoise,navg,parmin,parmax,wtikh)
        
        PARAMETERS
        - tissue_par: list/array of tissue parameters, in the following order:
                       tissue_par[0] = s0 (apparent proton density)
                       tissue_par[1] = fv (vascular signal fraction, ranging in 0-1)                            
                       tissue_par[2] = Dv (apparent diffusion coefficient of vascular water, in s/mm2)
                       tissue_par[3] = T2v (T2 of vascular water, in ms)                      
                       tissue_par[4] = Dt (apparent diffusion coefficient of tissue, in s/mm2)
                       tissue_par[5] = Kt (apparent excess kurtosis coefficient of tissue, dimensionless)
                       tissue_par[6] = T2t (T2 of tissue, in ms)   
        - mri_seq: 2D matrix, with 1st row listing b-values (in s/mm^2) and 2nd row listing TE (in ms)
        - meas: list/array of measurements
        - sgmnoise: noise standard deviation, to be used for offset Gaussian likelihood maximisation. 
                    Use sgmnoise = 0.0 for pure Gaussian noise.
        - navg: number of signal averages for more accurate estimation of the noise floor from sgmnoise.
                Use navg = 1 if signal averaging was not performed (NSA = 1 or NEX = 1, depending on
                the vendor; note that if trace-DWI imaging was used, navg should be equal to 3). 
                If sgmnoise = 0.0, navg has no effect
        - parmin: array of lower bounds for tissue parameters (order: s0, fv, Dv, T2v, Dt, Kt, T2t); 
                 the true S0 bound will be obtained multiplyig parmin[0] by max(signal) 
        - parmax: array of upper bounds for tissue parameters (order: s0, fv, Dv, T2v, Dt, Kt, T2t); 
                 the true S0 bound will be obtained multiplyig parmax[0] by max(signal) 
        - wtikh: Tikhonov regularisation weight (must be >=0; set it to 0 for no regulsarisation.
                 Note that an effective regularisation weight value will depend on the range on which the MRI signal 
                 is defined: for instance, for an MRI signal varying in the range [0; SMAX], resonable regularisation weights 
                 are of the order of approx SMAX/3)
            
        RETURNS
        - fobj: objective function according to a offset Gaussian noise model with Tikhonov regularisation: 
        
                            fobj = F1 + w*F2 with
                    
                            F1 = SUM_OVER_n( (measurement_n - sqrt(prediction_n^2 + noise_floor^2))^2 )
                            F2 = SUM_OVER_k( ( (p_k - pmin_k)/(pmax_k - pmin_k) )^2 )
                            
                 Above:
                 -- n is the measurement index
                 -- k is the tissue parameter index
                 -- w >= 0 is the Tikhonov regularisation weight
                 -- signal predictions are obtained using the signal model implemented by function signal_gen()
                 -- noise_floor =  1.253*sgmnoise*np.sqrt(navg)
                 -- pmin_k is the lower bound for the k-th tissue parameter
                 -- pmax_k is the upper bound for the k-th tissue parameter
         
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology,Barcelona, Spain
               <fgrussu@vhio.net>
    
    GridSearch(mri_seq, meas, sgmnoise, navg, parmin=[0.8, 0.0, 4.0, 150.0, 0.2, 0.0, 20.0], parmax=[12.0, 1.0, 100.0, 250.0, 3.0, 5.0, 140.0], nword=4, wtikh=0.0)
        Grid searchto initialise regularised non-linear least square fitting of the T2-IVIM-Kurtosis model
        
        DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel
            
        INTERFACE
        tissue_estimate, fobj_grid = GridSearch(mri_seq,meas,nfloor,parmin=default_min,parmax=default_max,nword=4,wtikh=0.0)
        
        PARAMETERS
        - mri_seq: 2D matrix, with 1st row listing b-values (in s/mm^2) and 2nd row listing TE (in ms)
        - meas: list/array of measurements
        - sgmnoise: noise standard deviation, to be used for offset Gaussian likelihood maximisation. 
                    Use sgmnoise = 0.0 for pure Gaussian noise.
        - navg: number of signal averages for more accurate estimation of the noise floor from sgmnoise.
                Use navg = 1 if signal averaging was not performed (NSA = 1 or NEX = 1, depending on
                the vendor; note that if trace-DWI imaging was used, navg should be equal to 3). 
                If sgmnoise = 0.0, navg has no effect
        - parmin: lower bound for tissue parameters. Default parmin=default_min, with default_min
                  elements being (in order):
                       S0 = 0.8 for apparent proton density (true S0 bound obtained multiplyig this number of max(signal))
                       fv = 0.0 for vascular signal fraction                           
                       Dv = 4 um2/ms for the vascular apparent diffusion coefficient
                       T2v = 150 ms for the vascular T2                    
                       Dt = 0.2 um2/ms for the tissue apparent diffusion coefficient
                       Kt = 0.0 for the tissue apparent excess kurtosis coefficient
                       T2t = 20.0 for the tissue T2
        - parmax: upper bound for tissue parameters. Default parmax=default_max, with default_max
                  elements being (in order):
                       S0 = 12.0 for apparent proton density (true S0 bound obtained multiplyig this number of max(signal))
                       fv = 1.0 for vascular signal fraction                        
                       Dv = 100 um2/ms for the vascular apparent diffusion coefficient
                       T2v = 250 ms for the vascular T2                    
                       Dt = 3.0 um2/ms for the tissue apparent diffusion coefficient
                       Kt = 5.0 for the tissue apparent excess kurtosis coefficient
                       T2t = 140.0 for the tissue T2        
        - nword: number of values to test for each tissue parameter in the grid search. Default nword=4
        - wtikh: Tikhonov regularisation weight (>=0). Default 0.0, implying no regularisation. 
                 Note that an effective regularisation weight value will depend on the range on which the MRI signal 
                 is defined: for instance, for an MRI signal varying in the range [0; SMAX], resonable regularisation weights 
                 are of the order of approx SMAX/3)
            
        RETURNS
        - tissue_estimate: array of tissue parameter that minimise the objective function Fobj(), obtained from 
                           a discrete list of tissue parameter candidates (i.e., the grid). Parameters are
                           tissue_estimate[0] = S0 (apparent proton density)
                           tissue_estimate[1] = fv (vascular signal fraction, ranging in 0-1)                            
                           tissue_estimate[2] = Dv (apparent diffusion coefficient of vascular water, in s/mm2)
                           tissue_estimate[3] = T2v (T2 of vascular water, in ms)                      
                           tissue_estimate[4] = Dt (apparent diffusion coefficient of tissue, in s/mm2)
                           tissue_estimate[5] = Kt (apparent excess kurtosis coefficient of tissue, dimensionless)
                           tissue_estimate[6] = T2t (T2 of tissue, in ms)   
        - fobj_grid:       value of the minimised objective function  Fobj()
         
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain
               <fgrussu@vhio.net>
    
    signal_gen(mri_seq, tissue_par)
        Function synthesising the MRI signals for the T2-IVIM-Kurtosis model
        
        DEPENDENCIES: argparse, os, sys, multiprocessing, time (python standard library); numpy, scipy and nibabel          
            
        INTERFACE
        signal = signal_gen(mri_seq,tissue_par)
        
        PARAMETERS
        - mri_seq: 2D matrix, with 1st row listing b-values (in s/mm^2) and 2nd row listing TE (in ms)
        - tissue_par: list/array of tissue parameters, in the following order:
                       tissue_par[0] = s0 (apparent proton density)
                       tissue_par[1] = fv (vascular signal fraction, ranging in 0-1)                            
                       tissue_par[2] = Dv (apparent diffusion coefficient of vascular water, in um2/ms)
                       tissue_par[3] = T2v (T2 of vascular water, in ms)                      
                       tissue_par[4] = Dt (apparent diffusion coefficient of tissue, um2/ms)
                       tissue_par[5] = Kt (apparent excess kurtosis coefficient of tissue, dimensionless)
                       tissue_par[6] = T2t (T2 of tissue, in ms)                    
        RETURNS
        - signal: a numpy array of measurements generated according to the model
                    
                     S = S0*(fv*exp(-b Dv - TE/T2v) + (1 - fv)*exp( -b Dt + (Kt/6)*(b Dt)^2 - TE/T2t)
         
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona, Spain
               <fgrussu@vhio.net>

```
