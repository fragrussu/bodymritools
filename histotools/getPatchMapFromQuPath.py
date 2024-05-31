### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

import argparse, os, sys
import numpy as np
import nibabel as nib
import pandas
import scipy.stats as st
import pickle as pk


def GetPatchWiseMaps(flist,H,W,Hp,Wp,Zp,out,SzMax=35.0):
	'''
	This program converts a CSV file storing information on cell detection from high-resolution optical imaging of
	stained histological sections to a patch-wise parametric maps for comparison with Magnetic Resonance Imaging (MRI). 
	Author: Francesco Grussu, Vall d Hebron Institute of Oncology (<fgrussu@vhio.net><francegrussu@gmail.com>). 
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

	'''
	### Load CSV files and create an array of cell areas and cell positions
	for nn in range(0,len(flist)):
		mycsv = flist[nn]
		print('')
		print('     ... loading file {} and stacking cell sizes and cell positions ...'.format(mycsv))
		d = pandas.read_csv(mycsv)
		if nn==0:
			area_array = d['Cell: Area']
			Wpos_array = d['Centroid X µm']
			Hpos_array = d['Centroid Y µm']
			OD_array = d['Cell: Eosin OD mean']
		else:
			area_array = np.concatenate([area_array,d['Cell: Area']])
			Wpos_array = np.concatenate([Wpos_array,d['Centroid X µm']]) 
			Hpos_array = np.concatenate([Hpos_array,d['Centroid Y µm']])
			OD_array = np.concatenate([OD_array,d['Cell: Eosin OD mean']])
		del d
	
	area_array = np.array(area_array)
	Wpos_array = np.array(Wpos_array)
	Hpos_array = np.array(Hpos_array)
	OD_array = np.array(OD_array)
	
	# Normalise optical density so that it ranges between 0 and 1
	OD_array = ( OD_array - np.min(OD_array) )/( np.max(OD_array) - np.min(OD_array) )
	
	### Process all cells and fill out a patchwise map
	
	## Get number of cells
	Ncells = area_array.size
	
	## Create empty patch-wise maps
	NHp = np.ceil(H/Hp)    # Number of patches along the image height
	NWp = np.ceil(W/Wp)    # Number of patches along the image width
	Hcorr = NHp*Hp         # Corrected histological image height after ceiling
	Wcorr = NWp*Wp         # Corrected histological image width after ceiling
	print('')
	print('     ... creating patch information:')
	print('                                    - {} patches of size {} um along image width (size: {} um)'.format(NWp,Wp,Wcorr))
	print('                                    - {} patches of size {} um along image heigth (size: {} um)'.format(NHp,Hp,Hcorr))
	vwLmap = np.zeros((int(NHp),int(NWp)))      # Allocate volume-weighted cell size map
	avgLmap = np.zeros((int(NHp),int(NWp)))   # Allocate arithmetic mean cell size map
	stdLmap = np.zeros((int(NHp),int(NWp)))   # Allocate cell size standard deviation map
	skewLmap = np.zeros((int(NHp),int(NWp)))   # Allocate cell size skewness map
	Cmap = np.zeros((int(NHp),int(NWp)))   # Allocate cellularity map in cells/mm2
	Fmap = np.ones((int(NHp),int(NWp)))    # Allocate intra-cellular patch fraction map
	ODmap = np.ones((int(NHp),int(NWp)))  # Allocate optical density of eosin
	
	## Find center of patches
	Hp_edges = np.linspace(0,Hcorr,int(NHp)+1) 
	Hp_centres = Hp_edges[0:int(NHp)] + Hp/2.0   # Position of the centre of each patch
	Wp_edges = np.linspace(0,Wcorr,int(NWp)+1) 
	Wp_centres = Wp_edges[0:int(NWp)] + Wp/2.0   # Position of the centre of each patch
	
	## Assign a label to all patches
	labelmap = np.zeros((int(NHp),int(NWp)))     # Map storing the label of each patch
	hhmap = np.zeros((int(NHp),int(NWp)))        # Map storing the patch number along the image heigth
	wwmap = np.zeros((int(NHp),int(NWp)))        # Map storing the patch number along the image width
	patch_count = 1
	for hh in range(0,int(NHp)):
		for ww in range(0,int(NWp)):
			labelmap[hh,ww] = patch_count
			hhmap[hh,ww] = hh
			wwmap[hh,ww] = ww
			patch_count = patch_count + 1
	
	## Assign cells to patches
	print('')
	print('     ... assigning cells to patches ...')
	cell_label = np.zeros(Ncells)
	for cc in range(0,Ncells):
	
		# Get cell position
		cc_hpos = Hpos_array[cc]
		cc_wpos = Wpos_array[cc]
		
		# Find patch to which the cell belongs
		ww_found = np.argmin( np.abs(Wp_centres - cc_wpos) )
		hh_found = np.argmin( np.abs(Hp_centres - cc_hpos) )
		
		# Store the label of the patch 
		cell_label[cc] = labelmap[hh_found,ww_found]	
			
	### Process patches
	print('')
	print('     ... calculating patch-wise statistics ...')
	
	# Create list to store cell size in each patch
	cell_list = [[]]*int(NHp)
	for hh in range(0,int(NHp)):
		cell_list[hh] = [[]]*int(NWp)
	
	# Loop through patches
	for hh in range(0,int(NHp)):
		for ww in range(0,int(NWp)):
			
			# Get the label of current patch
			hh_ww_label = labelmap[hh,ww]
			
			# Get histology stats from current patch
			hh_ww_OD = OD_array[cell_label==hh_ww_label]                                  # Array of optical densities of eosin
			hh_ww_A = area_array[cell_label==hh_ww_label]                                 # Array of areas in um^2
			hh_ww_L = (2.0/np.sqrt(np.pi))*np.sqrt(hh_ww_A)                               # Array of sizes in um
			hh_ww_A[hh_ww_L>SzMax] = np.nan                                               # Remove unrealistically large cells in cell area array
			hh_ww_L[hh_ww_L>SzMax] = np.nan                                               # Remove unrealistically large cells in cell size array
			cell_list[hh][ww] = hh_ww_L                                                   # Store array of sizes in um for the current patch
			
			# Compute metrics: volume-weighted cell size, and moments of the cell size distribution 
			hh_ww_Lmri = ( np.nanmean(hh_ww_L**7)/np.nanmean(hh_ww_L**3) )**(1/4)         # Get patch-wise volume-weighted CSI map in um
			hh_ww_avgLmri = np.nanmean(hh_ww_L)                                           # Get arithmetic cell size map in um
			hh_ww_stdLmri = np.nanstd(hh_ww_L)                                            # Get standard deviation of cell size map in um
			hh_ww_skewLmri = st.skew(hh_ww_L)                                             # Get skewness of cell size map in um
			
			# Compute metrics: cellularity in cells/mm2
			hh_ww_Ncells = np.sum(~np.isnan(hh_ww_A))                                     # Number of cells within patch
			if (np.isnan(hh_ww_Lmri)):
				hh_ww_cell = 0.0
			else:
				hh_ww_cell = hh_ww_Ncells/(1e-6*Hp*Wp)         # Get cellularity in cells/mm2 as C = number_cells_within_patch / patch_area
			
			# Compute metrics: intra-cellular patch fraction
			hh_ww_iFrac = np.nansum(hh_ww_A)/(Hp*Wp)               # Get intra-cellular fraction as Fc = sum(cell_area) / patch_area
			if (np.isnan(hh_ww_iFrac)):
				hh_ww_iFrac = 0.0
			if (hh_ww_iFrac>1.0):
				hh_ww_iFrac = 1.0
								
			# Compute metrics: optical mean optical density of eosin	
			hh_ww_ODmean = 1.0 - np.nansum( (1.0 - hh_ww_OD)*hh_ww_A)/np.nansum(hh_ww_A)  # Get mean optical density of eosin
			if(np.isnan(hh_ww_ODmean)):
				hh_ww_ODmean = 1.0
			
			# Save in the 2D patch-wise maps
			vwLmap[hh,ww] = hh_ww_Lmri
			avgLmap[hh,ww] = hh_ww_avgLmri
			stdLmap[hh,ww] = hh_ww_stdLmri
			skewLmap[hh,ww] = hh_ww_skewLmri
			Cmap[hh,ww] = hh_ww_cell
			Fmap[hh,ww] = hh_ww_iFrac
			ODmap[hh,ww] = hh_ww_ODmean
	

	
	### Save output patch-wise maps
	print('')
	print('     ... saving output patch-wise maps ...')
	
	## As python binaries
	np.save('{}_vwLum.npy'.format(out),vwLmap)                  # Cell size index as Numpy binary
	np.save('{}_avgLum.npy'.format(out),avgLmap)               # Arithmetic mean cell size as Numpy binary
	np.save('{}_stdLum.npy'.format(out),stdLmap)               # Cell size standard deviation as Numpy binary
	np.save('{}_skewLum.npy'.format(out),skewLmap)             # Cell size skewness as Numpy binary
	np.save('{}_Cellsmm2.npy'.format(out),Cmap)                # Cellularity in cells/mm2 as Numpy binary
	np.save('{}_FCellPatch.npy'.format(out),Fmap)              # Intra-cellular fraction as Numpy binary
	np.save('{}_ODeosin.npy'.format(out),ODmap)                # Optical density of eosin
	h_file = open('{}_CellSizePatches.bin'.format(out),'wb')   # List of cell sizes for all patches 
	pk.dump(cell_list,h_file,pk.HIGHEST_PROTOCOL)             
	h_file.close()	
	
	## As NIFTIs
	# Create header affine
	Amat = np.eye(4)
	Amat[0,0] = Hp/1000.0
	Amat[1,1] = Wp/1000.0
	Amat[2,2] = Zp/1000.0

	# Create 3D data matrices and save them: volume-weighted cell size index
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = vwLmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_vwLum.nii'.format(out))
	
	# Create 3D data matrices and save them: arithmetic mean cell size index
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = avgLmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_avgLum.nii'.format(out))

	# Create 3D data matrices and save them: cell size standard deviation index
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = stdLmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_stdLum.nii'.format(out))

	# Create 3D data matrices and save them: cell size skewness index
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = skewLmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_skewLum.nii'.format(out))
		
	# Create 3D data matrices and save them: cellularity in cells/mm2
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = Cmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_Cellsmm2.nii'.format(out)) 
	
	# Create 3D data matrices and save them: intra-cellular patch fraction
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = Fmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_FCellPatch.nii'.format(out))  
	
	# Create 3D data matrices and save them: eosin mean optical density
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = ODmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_ODeosin.nii'.format(out))  

	

if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program converts a CSV file storing information on cell detection from high-resolution optical imaging of stained histological sections to a patch-wise parametric maps for comparison with Magnetic Resonance Imaging (MRI). Author: Francesco Grussu, Vall d Hebron Institute of Oncology (<fgrussu@vhio.net><francegrussu@gmail.com>). Copyright (c) 2021 Vall d Hebron Institute of Oncology (VHIO), Barcelona, Spain. All rights reserved.')
	parser.add_argument('csv_list', help='list of paths to CSV files storing the cell segmentation information (multiple files may be required for very large histological images). This code expect CSV files containing the following columns: -- column with variable name "Cell: Area", where different cell areas from all detected cells are reported in um^2, -- columns with variable names "Centroid X µm" and "Centroid Y µm", storing the position of cells (in um) along the X (horizontal, i.e. image width) and Y (vertical, i.e. image height) direction -- column with variable name "Cell: Eosin OD mean", storing the estimated mean eosin signal value per cell')
	parser.add_argument('Hfov', help='field-of-view along the vertical direction (i.e. image height, in um) of the source histological image on which cells where segmented')
	parser.add_argument('Wfov', help='field-of-view along the horizontal direction (i.e. image width, in um) of the source histological image on which cells where segmented')
	parser.add_argument('Hpatch', help='height of the patches in um, along the vertical direction (i.e. along the image height), within which statistics of cell size will be calculated. It should match the resolution along the same spatial direction of the MRI scan to which histological information is to be compared')
	parser.add_argument('Wpatch', help='width of the patches in um, along the horizontal direction (i.e. along the image width), within which statistics of cell size will be calculated. It should match the resolution along the same spatial direction of the MRI scan to which histological information is to be compared')
	parser.add_argument('Zpatch', help='thickness of the MRI slice to which the 2D histology is to be compared to (used to create the NIFTI header)')
	parser.add_argument('out_base', help='root name of output files. There will be 5 output NIFTI files, with the following string added to the root name: *_vwLum.nii -> volume-weighted cell size index (CSI), in um, with CSI = (<L^7>/<L^3>)^1/4, where L is the size (apparent diameter) of individual cells within a patch, as shown in Grussu F et al, Magnetic Resonance in Medicine 2022, 88(1): 365-379, doi: 10.1002/mrm.29174 *_avgLum.nii -> mean cell size (arithmetic mean), in um, i.e. <L>,  where L is the size (apparent diameter) of individual cells within a patch, *_stdLum.nii -> cell size standard deviation, in um, i.e. sqrt( var(L) ),  where L is the size (apparent diameter) of individual cells within a patch,  *_skewLum.nii -> skewness of cell size distribution,  i.e. skew(L), where L is the size (apparent diameter) of individual cells within a patch, and where skew() is the Fisher-Pearson coefficient of skewness as implemented in Scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html)')
	parser.add_argument('--szmax', metavar='<value>', default='28.0', help='maximum realistic cell size in um (default: 28 um; cells larger than this value will be ignored)')
	
	args = parser.parse_args()

	### Get input information
	instr = args.csv_list
	inlist = instr.split(',')
	Hfov = float(args.Hfov)
	Wfov = float(args.Wfov)
	Hpatch = float(args.Hpatch)
	Wpatch = float(args.Wpatch)
	Zpatch = float(args.Zpatch)
	szmax = float(args.szmax)
	out_base = args.out_base
	
	### Print feedback
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('                             getPatchMapFromQuPath.py                               ')
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('')
	print('')
	print('* List of CSV files to process: {}'.format(inlist))
	print('* Source histological image height: {} um'.format(Hfov))
	print('* Source histological image width: {} um'.format(Wfov))
	print('* Target patch height: {} um'.format(Hpatch))
	print('* Target patch width: {} um'.format(Wpatch))
	print('* Target MRI slice thickness: {} um'.format(Zpatch))
	print('* Maximum allowed cell size: {} um'.format(szmax))
	print('* Output NIFTI files: {}_vwLum.nii, {}_avgLum.nii, {}_stdLum.nii, {}_skewLum.nii, {}_Cellsmm2.nii, {}_FCellPatch.nii, {}_ODeosin.nii'.format(args.out_base, args.out_base, args.out_base, args.out_base, args.out_base, args.out_base, args.out_base))
	print('* Output binary files: {}_vwLum.npy, {}_avgLum.npy, {}_stdLum.npy, {}_skewLum.npy, {}_Cellsmm2.npy, {}_FCellPatch.npy, {}_ODeosin.npy, {}_CellSizePatches.bin'.format(args.out_base, args.out_base, args.out_base, args.out_base, args.out_base, args.out_base, args.out_base, args.out_base))
	print('')
	print('')
	
	### Run code
	GetPatchWiseMaps(inlist,Hfov,Wfov,Hpatch,Wpatch,Zpatch,out_base,SzMax=szmax)
		
	### Done
	print('')
	print('     ... Done')
	print('')

