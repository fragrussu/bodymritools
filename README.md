# bodymritools

This repository contains several scripts and routines that one may find useful for the analysis of Magnetic Resonance Imaging (MRI) - mainly diffusion MRI (dMRI) - and histological data in body applications, as for example in liver imaging. The tools have been developed by Francesco Grussu as part of a **”la Caixa” Foundation** Junior Leader fellowship at the **Vall d'Hebron Institute of Oncology (VHIO)**, Barcelona (Spain). Contact: <fgrussu@vhio.net>.

**The project that gave rise to these results received the support of a fellowship from ”la Caixa” Foundation (ID 100010434). The fellowship code is "LCF/BQ/PR22/11920010"**. 

If you use _**bodymritools**_ for your research, please cite our preprint:

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
Tools that enable fitting several MRI signal models or representations (mainly dMRI) from scans in NIFTI format and sequence parameters stored as text files. These are:
* [**getAdcAkc.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/getAdcAkc.py): tool fitting a mono-dimensional _Diffusion Kurtosis Imaging_ signal representation (Jensen JH et al, Magn Res Med 2005, 53(6): 1432-1440, doi: [10.1002/mrm.20508](https://doi.org/10.1002/mrm.20508)). Manual here.
* [**getT2ivimkurtReg.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/getT2ivimkurtReg.py): tool fitting a two-pool, diffusion-T2 relaxation model disentangling tissue vs vascular signals. The model, which we refer to as _T2-IVIM-Kurtosis_, extends the approach of Jerome NP et al, Phys Med Biol 2016, 61(24): N667–N680, doi: [10.1088/1361-6560/61/24/N667](https://doi.org/10.1088/1361-6560/61/24/N667) to account for non-Gaussian diffusion in the tissue signal component, through a diffusion excess kurtosis parameter. Manual here.
* [**pgse2sphereinex.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/pgse2sphereinex.py): tool fitting a two-compartment model of intra-cellular, restricted diffusion within spherical cells, and extra-cellular hindered diffusion. Popular techniques such as VERDICT (Panagiotaki E et al, 
Cancer Res 2014, 74(7): 1902-1912, doi: [10.1158/0008-5472.CAN-13-2511](https://doi.org/10.1158/0008-5472.CAN-13-2511)) and IMPULSED (Jiang X et al, Magn Res Med 2016, 75(3): 1076-1085, doi: [10.1002/mrm.25684](https://doi.org/10.1002/mrm.25684)) are practical implementations of this general model. They feature specific assumptions on the values of the intrinsic diffusivities, the functional form of the extra-cellular diffusion process, and can account for vascular signals. In our tools, an option enables the selection of various, practical implementations of the intra-/extra-cellular two-pool model. We compared them systematically in our preprint Grussu F et al, medRxiv 2024: 2024.04.26.24306429, doi: [10.1101/2024.04.26.24306429](https://doi.org/10.1101/2024.04.26.24306429) (i.e., models _Diff-in_, _Diff-in-ex_, _Diff-in-exFast_, _Diff-in-exTD_, _Diff-in-exTDFast_ - see preprint for details). Manual here.
* [**mri2micro_dictml.py**](https://github.com/fragrussu/bodymritools/blob/main/mrifittools/mri2micro_dictml.py): tool fitting any desired MRI signal model given examples of simulated signals and corresponding tissue parameters. These are used to learn a forward signal representations mapping tissue parameters to the MRI signal for the specific input acquisition protocol. This representation is then plugged into standard maximum-likelihood fitting. The approach was used in Grussu F et al, medRxiv 2024: 2024.04.26.24306429, doi: [10.1101/2024.04.26.24306429](https://doi.org/10.1101/2024.04.26.24306429) for dMRI acquisitions for which analytical signal expressions are not readily available, using Monte Carlo simulations to generate the example of signals and tissue parameters. Manual here.

## _histotools_ scripts
Tools facilitating the comparison of histological data to MRI scans and parameteric maps.
* [**getPatchMapFromQuPath.py**](https://github.com/fragrussu/bodymritools/blob/main/histotools/getPatchMapFromQuPath.py): tool transforming a list of cell detection measurements obtained from [QuPath](https://qupath.github.io) into a patch-wise 2D NIFTI map. The computation of patch-wise maps from histology is demonstrated, for example, in Grussu F et al, Magn Res Med 2022, 88(1): 365-379, doi: [10.1002/mrm.29174](https://doi.org/10.1002/mrm.29174).

## _misctools_ scripts
Tools that MRI researchers may find useful in their day-to-day life. For example:
* [**segnrrd2nii.py**](https://github.com/fragrussu/bodymritools/blob/main/misctools/segnrrd2nii.py): a tool converting a segmentation drawn in [3D Slicer](https://www.slicer.org) and stored in .nrrd format, into a .nii segmentation, given a reference NIFTI file.

