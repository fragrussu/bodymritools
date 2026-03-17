This script can be called directly from the command line. Its input parameters are the following:
```
usage: pgse2sphmean.py [-h] dwi scheme out
Calculation of the spherical mean signal for a Pulsed-Gradient Spin Echo (PGSE) experiment. The script is
intended for use on a data acquired with PGSE where multiple gradient directions are used to acquire images
at varying (b,δ,Δ). The script will output the mean of all volumes acquired at fixed (b,δ,Δ). Author:
Francesco Grussu, Vall d Hebron Institute of Oncology fgrussu@vhio.net. Copyright (c) 2026, Fundació
Privada Institut d Investigació Oncològica de Vall d Hebron (Vall d Hebron Institute of Oncology, VHIO,
Barcelona, Spain)
positional arguments:

dwi         path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values
            and, potentially, multiple diffusion times

scheme      path of a text file storing the sequence parameters as a space-separated text file, made of 3
            rows x M columns, where column m = 1, ..., M corresponds to the m-th volume of the input NIFTIfile. 
            -- First row: b-values in s/mm2; 
            -- Second row: gradient duration small delta in ms; 
            -- Third row: gradient separation Large Delta in ms

out         output: root file name of output files; output NIFTIs will be stored as double-precision
            floating point images (FLOAT64), and the file names will end in: 
               *_SphMean.nii.gz: 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation
               *_SphMean.acq.txt: space-separated text file storing the sequence parameters correspnding to *_SphMean.nii.gz. 
                                  It features the same number of columns as *_SphMean.nii, and has 3 lines: 
                                  -- First row: b-values in s/mm2 
                                  -- Second row: gradient duration small delta (δ) in ms 
                                  -- Third row: gradient separation Large Delta (Δ) in ms.

options:
-h, --help  show this help message and exit

```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
NAME
pgse2sphmean
DESCRIPTION
### Copyright (c) 2026, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain).
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license (CC BY-NC-SA 4.0).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.
FUNCTIONS
run(mrifile, mriseq, output)
Calculation of the spherical mean signal for a Pulsed-Gradient Spin Echo (PGSE) experiment.
The script is intended for use on a data acquired with PGSE where multiple gradient directions are used to
acquire images at varying (b,δ,Δ). The script will output the mean of all volumes acquired at fixed (b,δ,Δ)

        USAGE
        run(mrifile, mriseq, output)

        * mrifile:    path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values and/or diffusion times

        * mriseq:     path of a text file storing the sequence parameters as a space-separated text file, made of 3 rows x M columns,
                                      where column m = 1, ..., M corresponds to the m-th volume of the input NIFTI file.
                                      -- First row: b-values in s/mm2
                                      -- Second row: gradient duration small delta (δ) in ms
                                      -- Third row: gradient separation Large Delta (Δ) in ms

        * output:     root file name of output files; output NIFTIs will be stored as double-precision floating point images
                      (FLOAT64), and the file names will end in:
                      *_SphMean.nii.gz: 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation
                      *_SphMean.acq.txt: space-separated text file storing the sequence parameters correspnding to *_SphMean.nii.gz. It features
                                         the same number of columns as *_SphMean.nii, and has 3 lines:
                                         -- First row: b-values in s/mm2
                                         -- Second row: gradient duration small delta (δ) in ms
                                         -- Third row: gradient separation Large Delta (Δ) in ms

        Third-party dependencies: nibabel, numpy
        Last successful test with nibabel=5.3.3, numpy=2.4.1

        Author: Francesco Grussu, Vall d'Hebron Institute of Oncology
                <fgrussu@vhio.net>

        Copyright (c) 2026, Fundació Privada Institut d'Investigació Oncològica de Vall d'Hebron
        (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain).
```
