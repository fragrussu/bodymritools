# bodymritools

This repository contains several scripts and routines that one may find useful for the analysis of Magnetic Resonance Imaging (MRI) - mainly diffusion MRI (dMRI) - and histological data in body applications, as for example in liver imaging. The tools have been developed by Francesco Grussu as part of a **”la Caixa” Foundation** Junior Leader fellowship at the **Vall d'Hebron Institute of Oncology (VHIO)**, Barcelona (Spain). Contact: <fgrussu@vhio.net>.
https://github.com/fragrussu/bodymritools/blob/main/logos/bodymritools.png

<div align="center">
  <img src="https://github.com/fragrussu/bodymritools/blob/main/logos/bodymritools.png" alt="MRHistoillustration" width="500" height="auto">
</div>

**The project that gave rise to these results received the support of a fellowship from ”la Caixa” Foundation (ID 100010434). The fellowship code is "LCF/BQ/PR22/11920010"**. 

If you use _bodymritools_ for your research, please cite our preprint:

*Francesco Grussu, Kinga Bernatowicz, Marco Palombo, Irene Casanova-Salas, Ignasi Barba, Sara Simonetti, Garazi Serna, Athanasios Grigoriou, Anna Voronova, Valezka Garay, Juan Francisco Corral, Marta Vidorreta, Pablo Garcia-Polo Garcia, Xavier Merino, Richard Mast, Nuria Roson, Manuel Escobar, Maria Vieito, Rodrigo Toledo, Paolo Nuciforo, Joaquin Mateo, Elena Garralda, Raquel Perez-Lopez*. **"Histology-informed liver diffusion MRI: biophysical model design and demonstration in cancer immunotherapy"**.  medRxiv 2024: 2024.04.26.24306429, doi: [10.1101/2024.04.26.24306429](https://doi.org/10.1101/2024.04.26.24306429). 
<div align="center">
  <img src="https://github.com/fragrussu/bodymritools/blob/main/logos/medrxivqr.png" alt="QRpreprint" width="100" height="auto">
</div>

## License
This repository is distributed under the **Attribution-NonCommercial-ShareAlike 4.0 International license** ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)). Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology (VHIO), Barcelona, Spain). All rights reserved. Link to license [here](https://github.com/fragrussu/bodymritools/blob/main/LICENSE.txt). 

**The use of _bodymritools_ MUST also comply with the individual licenses of all of its dependencies.**

## Download and requirements
To get _bodymritools_, simply clone this GitHub repository:
```
git clone https://github.com/fragrussu/bodymritools
```

_bodymritools_ has been written in python 3.10.9 ([anaconda](https://www.anaconda.com) distribution). It relies on the following third-party packages: 
* [pandas](https://pandas.pydata.org) (developed with version '1.5.3')
* [nibabel](https://nipy.org/nibabel) (developed with version '5.1.0')
* [numpy](https://numpy.org) (developed with version '1.23.5')
* [scipy](https://scipy.org) (developed with version '1.10.1')

Third-party packages have been developed by different authors and have their own licenses. **The use of _bodymritools_ MUST also comply with the individual licenses of all of its dependencies.**


## Content
_bodymritools_ scripts are stored within the [mrifittools](https://github.com/fragrussu/bodymritools/tree/main/mrifittools), [histotools](https://github.com/fragrussu/bodymritools/tree/main/histotools) and [misctools](https://github.com/fragrussu/bodymritools/tree/main/misctools) folders. Each tool contains a detailed help manual. Manuals are stored in the [manuals](https://github.com/fragrussu/bodymritools/tree/main/manuals) folder. 


To print a script's manual on your terminal, simply navigate to the script and type:
```
python <SCRIPT_NAME> -h
```
For example:
```
cd mrifittools 
python pgse2sphereinex.py -h
```

## _mrifittools_ scripts
[_mrifittools_](https://github.com/fragrussu/bodymritools/tree/main/mrifittools) enable voxel-wise fitting of several MRI signal models or representations (mainly dMRI) from scans in NIFTI format, given sequence parameters stored as text files. These are:
* [**getAdcAkc.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/getAdcAkc.py): tool fitting a mono-dimensional _**Diffusion Kurtosis Imaging**_ signal representation (Jensen JH et al, Magn Res Med 2005, 53(6): 1432-1440, doi: [10.1002/mrm.20508](https://doi.org/10.1002/mrm.20508)) to Pulsed-Gradient Spin Echo (PGSE) dMRI data. Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/getAdcAkc.md).
* [**getT2ivimkurtReg.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/getT2ivimkurtReg.py): tool fitting the **_two-pool, diffusion-T2 relaxation_ "_T2-IVIM-Kurtosis_"** model, disentangling tissue vs vascular signals, to Pulsed-Gradient Spin Echo (PGSE) dMRI data. The model extends the approach of Jerome NP et al, Phys Med Biol 2016, 61(24): N667–N680, doi: [10.1088/1361-6560/61/24/N667](https://doi.org/10.1088/1361-6560/61/24/N667) to account for non-Gaussian diffusion in the tissue signal component, through a diffusion excess kurtosis parameter. Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/getT2ivimkurtReg.md).
* [**pgse2sphereinex.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/pgse2sphereinex.py): tool fitting a _**two-compartment model of intra-cellular, restricted diffusion within spherical cells, and extra-cellular hindered diffusion**_ to Pulsed-Gradient Spin Echo (PGSE) dMRI data. Popular techniques such as _**VERDICT**_ (Panagiotaki E et al, 
Cancer Res 2014, 74(7): 1902-1912, doi: [10.1158/0008-5472.CAN-13-2511](https://doi.org/10.1158/0008-5472.CAN-13-2511)) and _**IMPULSED**_ (Jiang X et al, Magn Res Med 2016, 75(3): 1076-1085, doi: [10.1002/mrm.25684](https://doi.org/10.1002/mrm.25684)) rely on a similar diffusion model. In our tool, an option enables the selection of various, practical implementations of the intra-/extra-cellular two-pool model. _**We compared them systematically in our preprint**_ Grussu F et al, medRxiv 2024: 2024.04.26.24306429, doi: [10.1101/2024.04.26.24306429](https://doi.org/10.1101/2024.04.26.24306429) (i.e., **models** _**Diff-in**_, _**Diff-in-ex**_, _**Diff-in-exFast**_, _**Diff-in-exTD**_, _**Diff-in-exTDFast**_ - see preprint for details). Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/pgse2sphereinex.md).
* [**pgse2cylperpinex.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/pgse2cylperpinex.py): tool fitting a _**two-compartment model similar to the above to Pulsed-Gradient Spin Echo (PGSE) dMRI data. Here, cylinders are used in place of spheres**_. No gradient direction modelling is done - it is assumed that the diffusion gradients are always perfectly perpendicular to the cylinder longitudinal axis. Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/pgse2cylperpinex.md).
* [**pgse2sandi.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/pgse2sandi.py): tool fitting the _**SANDI dMRI model for brain grey matter**_  to Pulsed-Gradient Spin Echo (PGSE) dMRI data, via regularised, non-linear maximum-likelihood fitting. Details on SANDI can be found in Palombo M et al, NeuroImage 2020, 215: 116835, doi: [10.1016/j.neuroimage.2020.116835](https://doi.org/10.1016/j.neuroimage.2020.116835). Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/pgse2sandi.md).
* [**mTE2maps.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/mTE2maps.py): tool fitting a _**mono-exponential T2star decay**_ on multi-gradient echo (MGE) signal measurements, or a two-pool or three-pool implementation of the _**Susceptibility Perturbation MRI technique**_ by Santiago I et al, Cancer Res 2019, 79(9): 2435-2444, doi: [10.1158/0008-5472.CAN-18-3682](https://doi.org/10.1158/0008-5472.CAN-18-3682). Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/mTE2maps.md).
* [**mri2micro_dictml.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/mri2micro_dictml.py): tool _**fitting any desired MRI signal model given examples of simulated signals and corresponding tissue parameters**_. These are used to learn a forward signal representation mapping tissue parameters to the MRI signal for the specific input acquisition protocol. This representation is then plugged into standard maximum-likelihood non-linear fitting. The approach was used in Grussu F et al, medRxiv 2024: 2024.04.26.24306429, doi: [10.1101/2024.04.26.24306429](https://doi.org/10.1101/2024.04.26.24306429), where examples of signals and tissue parameters were obtained through Monte Carlo diffusion simulations. Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/mri2micro_dictml.md).

## _histotools_ scripts
[_histotools_](https://github.com/fragrussu/bodymritools/tree/main/histotools) facilitate the comparison between histological and MRI data. These are:
* [**getPatchMapFromQuPath.py**](https://github.com/fragrussu/bodymritools/blob/main/histotools/getPatchMapFromQuPath.py): tool transforming a list of cell detection measurements obtained from [QuPath](https://qupath.github.io) into _**patch-wise 2D histological NIFTI maps**_. The computation of patch-wise maps from histology follows the methodology shown in Grussu F et al, Magn Res Med 2022, 88(1): 365-379, doi: [10.1002/mrm.29174](https://doi.org/10.1002/mrm.29174). The same metrics as in that paper are computed. Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/getPatchMapFromQuPath.md).
* [**getHistoDensity.py**](https://github.com/fragrussu/bodymritools/blob/main/histotools/getHistoDensity.py): tool similar to the above, but including the calculation of additional _**2D patch-wise histological metrics**_ (e.g., cell counts, etc), as detailed in the manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/getHistoDensity.md).
* [**warpPatchHisto2MRI.py**](https://github.com/fragrussu/bodymritools/blob/main/histotools/warpPatchHisto2MRI.py): tool _**co-registering the manual outline of a specimen drawn on one MRI slice and on patch-wise maps**_ provided by [getPatchMapFromQuPath.py](https://github.com/fragrussu/bodymritools/blob/main/histotools/getPatchMapFromQuPath.py) or [getHistoDensity.py](https://github.com/fragrussu/bodymritools/blob/main/histotools/getHistoDensity.py). This co-registration approach has been used in in Grussu F et al, Magn Res Med 2022, 88(1): 365-379, doi: [10.1002/mrm.29174](https://doi.org/10.1002/mrm.29174). Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/warpPatchHisto2MRI.md).

## _misctools_ scripts
[_misctools_](https://github.com/fragrussu/bodymritools/tree/main/misctools) may be useful to MRI researchers in their day-to-day life. These are:
* [**ArtworkResample.py**](https://github.com/fragrussu/bodymritools/blob/main/misctools/ArtworkResample.py): a tool _**resizing Bitmap images to a specific width in inches and resolution in DPI**_. Different journals typically require paper figures with sizes/resolutions that vary from journal to journal - this tool enables quick, painless conversion and resampling for the most common file formats (e.g., PNG, TIFF, JPEG, etc). Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/ArtworkResample.md).
* [**unbiasSphMeanDWI.py**](https://github.com/fragrussu/bodymritools/blob/main/misctools/ArtworkResample.py): a tool _**mitigating the noise floor bias in spherical mean (SM) diffusion-weighted signals**_, using the analytical expression of [Pieciak T. et al, Proc ISMRM 2022, p.3900](https://archive.ismrm.org/2022/3900.html), i.e., _SM_unbiased = SM - 0.5*sigma^2/SM_. Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/unbiasSphMeanDWI.md).
* [**segnrrd2nii.py**](https://github.com/fragrussu/bodymritools/blob/main/misctools/segnrrd2nii.py): a tool _**converting a NRRD segmentation to NIFTI**_. For example, the tool can be used to convert segments drawn in [3D Slicer](https://www.slicer.org) and stored in NRRD format (.nrrd) into a NIFTI (.nii) label mask, given a reference NIFTI file. Manual [here](https://github.com/fragrussu/bodymritools/blob/main/manuals/segnrrd2nii.md).

