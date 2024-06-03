This script can be called directly from the command line. Its input parameters are the following: 
```
usage: segnrrd2nii.py [-h] seg ref out csv

Convert a .seg.nrrd segmentation to NIFTI, borrowing the header from a reference NIFTI
file. Author: Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona (Spain),
April-May 2021. Email: <fgrussu@vhio.net>.

positional arguments:
  seg         Segmentation file in NRRD format (.seg.nrrd, as those provided by Slicer)
  ref         Reference scan in NIFTI format (a.k.a. "master scan" using Slicer
              nomenclature)
  out         Output file storing the segmentation converted to NIFTI format
  csv         Output file storing the segmentation labels as a CSV file

options:
  -h, --help  show this help message and exit
```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module segnrrd2nii:

NAME
    segnrrd2nii

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.
    #
    #   Convert a .seg.nrrd segmentation to NIFTI, borrowing the header from a reference NIFTI file
    #   AUTHOR
    #   Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona (Spain)
    #   Email: <fgrussu@vhio.net>
    #

FUNCTIONS
    convert(segfile, reffile, outfile, outcsv)
        Convert a .seg.nrrd segmentation to NIFTI, borrowing the header from a reference NIFTI file
        
        USAGE
        convert(segfile, reffile, outfile, outcsv)
        
        INPUT ARGUMENTS
        * segfile: Segmentation file in NRRD format (.seg.nrrd, as those provided by Slicer)
        * reffile: Reference scan in NIFTI format (a.k.a. "master scan" using Slicer nomenclature)
        * outfile: Output file storing the segmentation converted to NIFTI format
        * outcsv:  Output file storing the segmentation labels as a CSV file
        
        AUTHOR
        Francesco Grussu, Vall d Hebron Institute of Oncology, Barcelona (Spain), April-May 2021
        Email: <fgrussu@vhio.net>
```   
