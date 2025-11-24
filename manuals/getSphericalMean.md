This script can be called directly from the command line. Its input parameters are the following: 
```
usage: getSphericalMean.py [-h] dwi scheme out

Compute spherical mean signal for a PGSE acquisition. Third-party dependencies: nibabel, numpy, scipy. Last
successful test with nibabel=5.1.0, scipy=1.10.1, numpy=1.23.5. Francesco Grussu, Vall d Hebron Institute of
Oncology <fgrussu@vhio.net>. Copyright (c) 2024-2025, Vall d Hebron Institute of Oncology (VHIO), Barcelona,
Spain. All rights reserved.

positional arguments:
  dwi         path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values
              and, potentially, multiple diffusion times
  scheme      path of a text file storing the sequence parameters as a space-separated text file, made of 3
              rows x M columns, where column m = 1, ..., M corresponds to the m-th volume of the input NIFTI
              file. -- First row: b-values in s/mm2; -- Second row: gradient duration small delta in ms; --
              Third row: gradient separation Large Delta in ms
  out         root file name of output files. Two output files will be saved: a NIFTI file with the
              spherical mean signal (*.nii.gz) and a textfile with the corresponding sequence parameters
              (*acq.txt). Output *nii.gz: file 4D NIFTI file storing a series of spherical mean signals at
              fixed b, grad. dur, grad. separation. Output *acq.txt: space-separated text file storing the
              sequence parameters correspnding to the spherical means. It features the same number of
              columns as *nii.gz, and has 3 lines: -- First row: b-values in s/mm2 -- Second row: gradient
              duration small delta in ms -- Third row: gradient separation Large Delta in ms

options:
  -h, --help  show this help message and exit
```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module getSphericalMean:

NAME
    getSphericalMean

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain).
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

FUNCTIONS
    run(mrifile, mriseq, output)
        Get spherical mean signals for a PGSE acquisition

        USAGE
        run(mrifile, mriseq, output)

        * mrifile:    path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values and/or diffusion times

        * mriseq:     path of a text file storing the sequence parameters as a space-separated text file, made of 3 rows x M columns,
                                      where column m = 1, ..., M corresponds to the m-th volume of the input NIFTI file.
                                      -- First row: b-values in s/mm2
                                      -- Second row: gradient duration small delta in ms
                                      -- Third row: gradient separation Large Delta in ms

        * output:     root file name of output files. Two output files will be saved: a NIFTI file with the spherical mean signal (*.nii.gz) and a text
                          file with the corresponding sequence parameters (*acq.txt).
                      Output *nii.gz: file 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation
                      Output *acq.txt: space-separated text file storing the sequence parameters correspnding to the spherical means. It features
                                         the same number of columns as *nii.gz, and has 3 lines:
                                         -- First row: b-values in s/mm2
                                         -- Second row: gradient duration small delta in ms
                                         -- Third row: gradient separation Large Delta in ms

        Third-party dependencies: nibabel, numpy, scipy
        Last successful test with nibabel=5.1.0, scipy=1.10.1, numpy=1.23.5

        Author: Francesco Grussu, Vall d'Hebron Institute of Oncology
                <fgrussu@vhio.net>

        Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron
        (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain).

```
