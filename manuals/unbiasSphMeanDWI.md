This script can be called directly from the command line. Its input parameters are the following: 
```
usage: unbiasSphMeanDWI.py [-h] [--bval <file>] [--nsa <num>] dwi sigma out

Mitigate the noise floor bias in spherical mean diffusion-weighted signals using the
analytical expression of Pieciak T. et al, Proc ISMRM 2022, p.3900 (SM_unbiased = SM -
0.5*sigma^2/SM). Author: Francesco Grussu, July 2023. Email: <fgrussu@vhio.net>. Third-
party dependencies: numpy, nibabel. Developed with numpy 1.23.5 and nibabel 5.1.0.

positional arguments:
  dwi            path of a 4D NIFTI file storing diffusion MRI data. The NIFTI file can
                 either i) already contain spherical mean data at fixed b-value, or ii)
                 contain images acquired at different gradient directions; in this case,
                 the spherical mean will be computed. The default is i); to flag that the
                 input images corresponds to different gradient directions, pass a b-value
                 list via option --bval (see below).
  sigma          path of a 3D NIFTI file storing a voxel-wise noise map (standard
                 deviation of Gaussian/Rician noise). This can be obained, for example,
                 via Marchenko-Pastur Principal Component Analysis (Veraart J et al,
                 NeuroImage 2016).
  out            path of a 4D NIFTI file storing the unbiased spherical mean. If the input
                 file contained images corresponding to different gradient directions, as
                 flagged by passing a b-value file with option --bv, then the output file
                 will have fewer volumes than the inut file, as it will contain spherical
                 mean signals. In this case, additional text file, where "*.bval_sph_mean"
                 is appended to the output file name, will be saved, to store the b-values
                 of the spherical mean signals.

options:
  -h, --help     show this help message and exit
  --bval <file>  path a a text file storing b-values (FSL format; in s/mm2; one raw,
                 space-separated). If provided, the code will assume that the input NIFTI
                 file contains images corresponding to different gradient directions, so
                 that the spherical mean needs calculating, before unbiasing can be
                 performed. Spherical mean signals will be calcualated by averaging images
                 with the same b-value. Default: None (no b-value list provided, implying
                 that the input NIFTI file stores spherical mean signals)
  --nsa <num>    total number of signal averages performed on the image on which the noise
                 map was calculated (default: --nsa 1). For example, if 1 b-value is
                 acquired with NEX=2 using trace imaging (acquisition of 3 orthogonal
                 gradients, each with NEX=2, which are then averaged by the scanner), then
                 NSA = 6 (2 averages per gradients, with averaging of gradients performed
                 by the scanner).
```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module unbiasSphMeanDWI:

NAME
    unbiasSphMeanDWI

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

FUNCTIONS
    unbiasSM(dwi, noise, outfile, nsa=1, bv=None)
        Mitigate the noise floor bias in spherical mean diffusion-weighted signals.
        
        USAGE:
        unbiasSM(dwi,noise,outfile,nsa=1,bv=None)
        
        INPUT PARAMETERS
        
        * dwi      : path of a 4D NIFTI file storing diffusion MRI data. 
                     The NIFTI file can either 
                     - already contain spherical mean data 
                       at fixed b-value, or 
                     --  contain images acquired at different
                         gradient directions; in this case, the spherical mean will be 
                         computed. 
                     The default is -; to flag that the input images corresponds to 
                     different gradient directions, pass a b-value
                     list via optional input parameter bv (see below).
        
        * noise    :  path of a 3D NIFTI file storing a voxel-wise noise map
                      (standard deviation of Gaussian/Rician noise). This can be obained, 
                      for example, via Marchenko-Pastur Principal Component 
                      Analysis, as in Veraart J et al, NeuroImage 2016.
        
        * outfile  :  path of a 4D NIFTI file storing the unbiased spherical mean. 
                      If the input file contained images corresponding to different 
                      gradient directions, as flagged by passing a b-value file with
                      input bv, then the output file will have fewer volumes
                      than the inut file, as it will contain spherical mean signals. 
                      In this case, additional text file, where "*.bval_sph_mean" is appended 
                      to the output file name, will be saved, to store the b-values 
                      of the spherical mean signals.
        
        * nsa     :   total number of signal averages performed on the image on which 
                      the noise map was calculated. For example, if 1 b-value is acquired with 
                      NEX=2 using trace imaging (acquisition of 3 orthogonal gradients, 
                      each with NEX=2, which are then averaged by the scanner), then NSA = 6 
                      (2 averages per gradients, with averaging of gradients 
                      performed by the scanner). Default: nsa=1.
        
        * bv       :  path a a text file storing b-values (FSL format; in s/mm2; one raw, 
                      space-separated). If provided, the code will assume that the input NIFTI
                      file contains images corresponding to different gradient directions, so 
                      that the spherical mean needs calculating, before unbiasing can be 
                      performed. Spherical mean signals will be calcualated by averaging images
                      with the same b-value. Default: None, i.e., no b-value list provided, implying that 
                      the input NIFTI file stores spherical mean signals.
        
        
        This tools mitigated the noise floor bias using the analytical expression of 
        Pieciak T. et al, Proc ISMRM 2022, p.3900 (SM_unbiased = SM - 0.5*sigma^2/SM). 
        
        Author: Francesco Grussu, July 2023. 
        Email: <fgrussu@vhio.net>. 
        
        Third-party dependencies: numpy, nibabel. 
        Developed with numpy 1.23.5 and nibabel 5.1.0.
```   
