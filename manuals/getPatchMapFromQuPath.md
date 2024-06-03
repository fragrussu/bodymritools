This script can be called directly from the command line. Its input parameters are the following: 
```
usage: getPatchMapFromQuPath.py [-h] [--szmax <value>] csv_list Hfov Wfov Hpatch Wpatch Zpatch out_base

This program converts a CSV file storing information on cell detection from high-resolution optical imaging of stained
histological sections to a patch-wise parametric maps for comparison with Magnetic Resonance Imaging (MRI). Author:
Francesco Grussu, Vall d Hebron Institute of Oncology (<fgrussu@vhio.net>). Copyright (c) 2021 Vall d Hebron Institute
of Oncology (VHIO), Barcelona, Spain. All rights reserved.

positional arguments:
  csv_list         list of paths to CSV files storing the cell segmentation information (multiple files may be required
                   for very large histological images). This code expect CSV files containing the following columns: --
                   column with variable name "Cell: Area", where different cell areas from all detected cells are
                   reported in um^2, -- columns with variable names "Centroid X µm" and "Centroid Y µm", storing the
                   position of cells (in um) along the X (horizontal, i.e. image width) and Y (vertical, i.e. image
                   height) direction -- column with variable name "Cell: Eosin OD mean", storing the estimated mean
                   eosin signal value per cell
  Hfov             field-of-view along the vertical direction (i.e. image height, in um) of the source histological
                   image on which cells where segmented
  Wfov             field-of-view along the horizontal direction (i.e. image width, in um) of the source histological
                   image on which cells where segmented
  Hpatch           height of the patches in um, along the vertical direction (i.e. along the image height), within
                   which statistics of cell size will be calculated. It should match the resolution along the same
                   spatial direction of the MRI scan to which histological information is to be compared
  Wpatch           width of the patches in um, along the horizontal direction (i.e. along the image width), within
                   which statistics of cell size will be calculated. It should match the resolution along the same
                   spatial direction of the MRI scan to which histological information is to be compared
  Zpatch           thickness of the MRI slice to which the 2D histology is to be compared to (used to create the NIFTI
                   header)
  out_base         root name of output files. There will be 5 output NIFTI files, with the following string added to
                   the root name: *_vwLum.nii -> volume-weighted cell size index (CSI), in um, with CSI =
                   (<L^7>/<L^3>)^1/4, where L is the size (apparent diameter) of individual cells within a patch, as
                   shown in Grussu F et al, Magnetic Resonance in Medicine 2022, 88(1): 365-379, doi: 10.1002/mrm.29174
                   *_avgLum.nii -> mean cell size (arithmetic mean), in um, i.e. <L>, where L is the size (apparent
                   diameter) of individual cells within a patch, *_stdLum.nii -> cell size standard deviation, in um,
                   i.e. sqrt( var(L) ), where L is the size (apparent diameter) of individual cells within a patch,
                   *_skewLum.nii -> skewness of cell size distribution, i.e. skew(L), where L is the size (apparent
                   diameter) of individual cells within a patch, and where skew() is the Fisher-Pearson coefficient of
                   skewness as implemented in Scipy
                   (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html)

options:
  -h, --help       show this help message and exit
  --szmax <value>  maximum realistic cell size in um (default: 28 um; cells larger than this value will be ignored)
```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
Help on module getPatchMapFromQuPath:

NAME
    getPatchMapFromQuPath

DESCRIPTION
    ### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
    #   All rights reserved.
    #   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
    #   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

FUNCTIONS
    GetPatchWiseMaps(flist, H, W, Hp, Wp, Zp, out, SzMax=35.0)
        This program converts a CSV file storing information on cell detection from high-resolution optical imaging of
        stained histological sections to a patch-wise parametric maps for comparison with Magnetic Resonance Imaging (MRI). 
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology (<fgrussu@vhio.net>). 
        Copyright (c) 2024 Vall d Hebron Institute of Oncology (VHIO), Barcelona, Spain. All rights reserved.
        
        INTERFACE
        GetPatchWiseMaps(csvlist,H,W,Hp,Wp,out,SzMax=28.0)
        
        - flist: list of paths to CSV files storing the cell segmentation information (multiple files may be required for very large histological images). 
                   This code expect CSV files containing the following columns: 
                           -- column with variable name "Cell: Area", where different cell areas from all detected cells are reported in um^2, 
                           -- columns with variable names "Centroid X µm" and "Centroid Y µm", storing the position of cells (in um) along the X 
                              (horizontal, i.e. image width) and Y (vertical, i.e. image height) direction
                           -- column with variable name "Cell: Eosin OD mean", storing the estimated mean eosin signal value per cell
        - H:       field-of-view along the vertical direction (i.e. image height, in um) of the source 
                   histological image on which cells where segmented
        - W:       field-of-view along the horizontal direction (i.e. image width, in um) of the source 
                   histological image on which cells where segmented
        - Hp:      heigth of the patches in um, along the vertical direction (i.e. along the image height), within which statistics 
                   of cell size will be calculated. It should match the resolution along the same spatial direction of the MRI scan 
                   to which histological information is to be compared
        - Wp:      width of the patches in um, along the horizontal direction (i.e. along the image width), within which statistics 
                   of cell size will be calculated. It should match the resolution along the same spatial direction of the MRI scan 
                   to which histological information is to be compared
        - Zp:      thickness of the MRI slice to which the 2D histology is to be compared to (used to create the NIFTI header)
        - out:     root name of output files. There will be 5 output NIFTI files, with the following string added to the root name: 
                   *_vwLum.nii -> volume-weighted cell size index (CSI), in um, with CSI = (<L^7>/<L^3>)^1/4, 
                                where L is the size (apparent diameter) of individual cells within a patch,
                                as shown in Grussu F et al, Magnetic Resonance in Medicine 2022, 88(1): 365-379, doi: 10.1002/mrm.29174 
                   *_avgLum.nii -> mean cell size (arithmetic mean), in um, i.e. <L>, 
                                 where L is the size (apparent diameter) of individual cells within a patch, 
                   *_stdLum.nii -> cell size standard deviation, in um, i.e. sqrt( var(L) ), 
                                 where L is the size (apparent diameter) of individual cells within a patch, 
                   *_skewLum.nii -> skewness of cell size distribution,  i.e. skew(L), 
                                 where L is the size (apparent diameter) of individual cells within a patch, 
                                                 and where skew() is the Fisher-Pearson coefficient of skewness as implemented in Scipy
                                                 (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html)          
                   *_FCellPatch.nii -> intra-cellular patch fraction Fc;
                   *_Cellsmm2.nii -> cellularity map in cells/mm2, defined as number_of_cells_within_patch/patch_area_in_mm2;
                   *_ODeosin.nii ->  mean optical density of eosin.
                   The script will also store several pickle binary python files: 
                   *_CellSizePatches.bin, storing a list G where G[i][j] lists the sizes of all cells within patch (i,j); 
                   files ending as *_vwLum.npy, *_avgLum.npy, *_stdLum.npy, *_skewLum.npy, *_Cellsmm2.npy, *_FCellPatch.npy, and *_ODeosin.npy 
                   storing the same maps as in the corresponding *nii files, but as NumPy binaries  
        - SzMax:   maximum realistic cell size in um (default: 28 um; cells larger than this value will be ignored)
```   
