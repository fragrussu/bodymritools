This script can be called directly from the command line. Its input parameters are the following: 
```
usage: getHistoDensity.py [-h] csv_list Hfov Wfov Hpatch Wpatch Zpatch out_base

This program converts a CSV file storing information on cell detection from high-resolution optical imaging of
stained histological sections to patch-wise parametric maps of cell density, intra-cellular fraction and mean cell
area in NIFTI format. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (<fgrussu@vhio.net>). Copyright
(c) 2022 Vall d Hebron Institute of Oncology (VHIO), Barcelona, Spain. All rights reserved.

positional arguments:
  csv_list    list of paths to CSV files storing the cell detection information (multiple files may be required for
              very large histological images). This code expects CSV files containing at least three columns, whose
              variable names are "Centroid_X", "Centroid_Y" and "Area". These are assumed to storee the position of
              cells (in um) along the X (horizontal, i.e. image width) and Y (vertical, i.e. image height) direction,
              as well as cell area (in um2)
  Hfov        field-of-view along the vertical direction (i.e. image height, in um) of the source histological image
              on which cells where segmented
  Wfov        field-of-view along the horizontal direction (i.e. image width, in um) of the source histological image
              on which cells where segmented
  Hpatch      height of the patches in um, along the vertical direction (i.e. along the image height), within which
              statistics of cell size will be calculated. It should match the resolution along the same spatial
              direction of the MRI scan to which histological information is to be compared
  Wpatch      width of the patches in um, along the horizontal direction (i.e. along the image width), within which
              statistics of cell size will be calculated. It should match the resolution along the same spatial
              direction of the MRI scan to which histological information is to be compared
  Zpatch      thickness of slice in the output NIFTI files (used only to create the NIFTI header)
  out_base    root name of output files. There will be the following output files, with the following string added to
              the root name: *_cellno.nii -> NIFTI file storing the number of detected cells within each path;
              *_cellno.npy -> same content as *_cellno.nii, but stored as a binary python file that contains the map
              as a NumPy array; *_cellsmm2.nii -> NIFTI file storing the density of detected cells within patches of
              the given size, measured in cell/mm2; *_cellsmm2.npy -> same content as *_cellsmm2.nii, but stored as a
              binary python file that contains the map as a NumPy array; *_cellareaum2.nii -> NIFTI file storing the
              mean cell area within patches of the given size, measured in um2; *_cellareaum2.npy -> same content as
              *_cellaream2.nii, but stored as a binary python file that contains the map as a NumPy array;
              *_cellfract.nii -> NIFTI file storing the intra-cellular area fraction within patches of the given
              size, measured in um2; *_cellfract.npy -> same content as *_cellfract.nii, but stored as a binary
              python file that contains the map as a NumPy array

options:
  -h, --help  show this help message and exit
```
Additionally, you can load the script directly into your python code as a module. The module has been organised as follows:
```
NAME
    getHistoDensity

FUNCTIONS
    run(flist, H, W, Hp, Wp, Zp, out)
        This program converts a CSV file storing information on cell detection from high-resolution optical imaging of stained histological sections 
        to patch-wise parametric maps of cell density, intra-cellular fraction and mean cell area in NIFTI format.
        Author: Francesco Grussu, Vall d Hebron Institute of Oncology (<fgrussu@vhio.net>). 
        Copyright (c) 2022 Vall d Hebron Institute of Oncology (VHIO), Barcelona, Spain. All rights reserved.
        
        INTERFACE
        run(flist,H,W,Hp,Wp,Zp,out)
        
        - flist:   list of paths to CSV files storing the cell detection information (multiple files may be required for very large histological images). 
                   This code expects CSV files containing at least three columns, whose variable names are "Centroid_X", "Centroid_Y" and "Area". 
                   These are assumed to storee the position of cells (in um) along the X (horizontal, i.e. image width) and Y 
                   (vertical, i.e. image height) direction, as well as cell areas (in um2)
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
        - Zp:      thickness of slice in the output NIFTI files (used only to create the NIFTI header)
        - out:     root name of output files. There will be the following output files, with the following string added to the root name:
                    *_cellno.nii -> NIFTI file storing the number of detected cells within each path; 
                    *_cellno.npy -> same content as *_cellno.nii, but stored as a binary python file that contains the map as a NumPy array;    
                    *_cellsmm2.nii -> NIFTI file storing the density of detected cells within patches of the given size, measured in cell/mm2; 
                    *_cellsmm2.npy -> same content as *_cellsmm2.nii, but stored as a binary python file that contains the map as a NumPy array; 
                    *_cellareaum2.nii -> NIFTI file storing the mean cell area within patches of the given size, measured in um2; 
                    *_cellareaum2.npy -> same content as *_cellaream2.nii, but stored as a binary python file that contains the map as a NumPy array;                             
                    *_cellfract.nii -> NIFTI file storing the intra-cellular area fraction within patches of the given size, measured in um2; 
                    *_cellfract.npy -> same content as *_cellfract.nii, but stored as a binary python file that contains the map as a NumPy array
```
