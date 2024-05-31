import argparse, os, sys
import numpy as np
import nibabel as nib
import pandas
import pickle as pk
from matplotlib import pyplot as plt

def run(flist,H,W,Hp,Wp,Zp,out):
	'''
	This program converts a CSV file storing information on cell detection from high-resolution optical imaging of stained histological sections 
	to patch-wise parametric maps of cell density, intra-cellular fraction and mean cell area in NIFTI format.
	Author: Francesco Grussu, Vall d Hebron Institute of Oncology (<fgrussu@vhio.net><francegrussu@gmail.com>). 
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
	'''
	### Load CSV files and create an array of cell areas and cell positions
	for nn in range(0,len(flist)):
		mycsv = flist[nn]
		print('')
		print('     ... loading file {} and stacking cell sizes and cell positions ...'.format(mycsv))
		d = pandas.read_csv(mycsv)
		if nn==0:
			area_array = d.Area
			Wpos_array = d.Centroid_X
			Hpos_array = d.Centroid_Y
		else:
			area_array = np.concatenate([area_array,d.Area]) 
			Wpos_array = np.concatenate([Wpos_array,d.Centroid_X]) 
			Hpos_array = np.concatenate([Hpos_array,d.Centroid_Y])
		del d
	area_array = np.array(area_array)
	Wpos_array = np.array(Wpos_array)
	Hpos_array = np.array(Hpos_array)
	
	### Process all cells and fill out a patchwise map
	
	## Get number of cells
	Ncells = Wpos_array.size
	
	## Create empty patch-wise maps
	NHp = np.ceil(H/Hp)    # Number of patches along the image height
	NWp = np.ceil(W/Wp)    # Number of patches along the image width
	Hcorr = NHp*Hp         # Corrected histological image height after ceiling
	Wcorr = NWp*Wp         # Corrected histological image width after ceiling
	print('')
	print('     ... creating patch information:')
	print('                                    - {} patches of size {} um along image width (size: {} um)'.format(NWp,Wp,Wcorr))
	print('                                    - {} patches of size {} um along image heigth (size: {} um)'.format(NHp,Hp,Hcorr))
	Nmap = np.zeros((int(NHp),int(NWp)))   # Allocate cell count map in no_cells
	Cmap = np.zeros((int(NHp),int(NWp)))   # Allocate cellularity map in cells/mm2
	Amap = np.zeros((int(NHp),int(NWp)))   # Allocate cell area map in um2
	Fmap = np.zeros((int(NHp),int(NWp)))   # Allocate intra-cellular fraction map [normalised units]
	
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
			
			# Get number of cells within the patch
			hh_ww_list = Wpos_array[cell_label==hh_ww_label] 
			hh_ww_Ncells = np.sum(~np.isnan(hh_ww_list))
			hh_ww_cell = hh_ww_Ncells/(1e-6*Hp*Wp)         # Get cellularity in cells/mm2 as C = number_cells_within_patch / patch_area
			
			# Get cell areas
			hh_ww_areas = area_array[cell_label==hh_ww_label]
			hh_ww_A = np.nanmean(hh_ww_areas)                    # Get mean cell area in um2
			hh_ww_Atot = np.nansum(hh_ww_areas)                  # Get total cell area
			hh_ww_F = hh_ww_Atot/(Hp*Wp)                         # Get intra-cellular area fraction
			if(hh_ww_F>1.0):
				hh_ww_F = 1.0
				
			# Save in the 2D patch-wise maps
			Nmap[hh,ww] = hh_ww_Ncells
			Cmap[hh,ww] = hh_ww_cell
			Amap[hh,ww] = hh_ww_A
			Fmap[hh,ww] = hh_ww_F
	
	
	### Save output patch-wise maps
	print('')
	print('     ... saving output patch-wise maps ...')
	
	## As python binaries
	np.save('{}_cellno.npy'.format(out),Nmap)               # Cellularity in cells/mm2 as Numpy binary
	np.save('{}_cellsmm2.npy'.format(out),Cmap)               # Cellularity in cells/mm2 as Numpy binary
	np.save('{}_cellareaum2.npy'.format(out),Amap)            # Cell area in um2 as Numpy binary
	np.save('{}_cellfract.npy'.format(out),Fmap)              # Intra-cellular fraction as Numpy binary
	
	## As NIFTIs
	# Create header affine
	Amat = np.eye(4)
	Amat[0,0] = Hp/1000.0
	Amat[1,1] = Wp/1000.0
	Amat[2,2] = Zp/1000.0
		
	# Create 3D data matrices and save them: cellularity in cells/mm2
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = Nmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_cellno.nii'.format(out)) 
	
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = Cmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_cellsmm2.nii'.format(out)) 
	
	# Create 3D data matrices and save them: mean cell area in um2
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = Amap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_cellareaum2.nii'.format(out)) 
	
	# Create 3D data matrices and save them: intra-cellular patch fraction
	Dmat = np.zeros((int(NHp),int(NWp),1))
	Dmat[:,:,0] = Fmap
	img = nib.Nifti1Image(Dmat, Amat)
	img.header.set_xyzt_units(2)
	img.set_data_dtype('float64')
	img.set_data_dtype('float64')
	nib.save(img,'{}_cellfract.nii'.format(out)) 
		
	


	

if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='This program converts a CSV file storing information on cell detection from high-resolution optical imaging of stained histological sections to patch-wise parametric maps of cell density, intra-cellular fraction and mean cell area in NIFTI format. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (<fgrussu@vhio.net><francegrussu@gmail.com>). Copyright (c) 2022 Vall d Hebron Institute of Oncology (VHIO), Barcelona, Spain. All rights reserved.')
	parser.add_argument('csv_list', help='list of paths to CSV files storing the cell detection information (multiple files may be required for very large histological images). This code expects CSV files containing at least three columns, whose variable names are "Centroid_X", "Centroid_Y" and "Area". These are assumed to storee the position of cells (in um) along the X (horizontal, i.e. image width) and Y (vertical, i.e. image height) direction, as well as cell area (in um2)')
	parser.add_argument('Hfov', help='field-of-view along the vertical direction (i.e. image height, in um) of the source histological image on which cells where segmented')
	parser.add_argument('Wfov', help='field-of-view along the horizontal direction (i.e. image width, in um) of the source histological image on which cells where segmented')
	parser.add_argument('Hpatch', help='height of the patches in um, along the vertical direction (i.e. along the image height), within which statistics of cell size will be calculated. It should match the resolution along the same spatial direction of the MRI scan to which histological information is to be compared')
	parser.add_argument('Wpatch', help='width of the patches in um, along the horizontal direction (i.e. along the image width), within which statistics of cell size will be calculated. It should match the resolution along the same spatial direction of the MRI scan to which histological information is to be compared')
	parser.add_argument('Zpatch', help='thickness of slice in the output NIFTI files (used only to create the NIFTI header)')
	parser.add_argument('out_base', help='root name of output files. There will be the following output files, with the following string added to the root name: *_cellno.nii -> NIFTI file storing the number of detected cells within each path; *_cellno.npy -> same content as *_cellno.nii, but stored as a binary python file that contains the map as a NumPy array; *_cellsmm2.nii -> NIFTI file storing the density of detected cells within patches of the given size, measured in cell/mm2; *_cellsmm2.npy -> same content as *_cellsmm2.nii, but stored as a binary python file that contains the map as a NumPy array; *_cellareaum2.nii -> NIFTI file storing the mean cell area within patches of the given size, measured in um2; *_cellareaum2.npy -> same content as *_cellaream2.nii, but stored as a binary python file that contains the map as a NumPy array; *_cellfract.nii -> NIFTI file storing the intra-cellular area fraction within patches of the given size, measured in um2; *_cellfract.npy -> same content as *_cellfract.nii, but stored as a binary python file that contains the map as a NumPy array')
	
	args = parser.parse_args()

	### Get input information
	instr = args.csv_list
	inlist = instr.split(',')
	Hfov = float(args.Hfov)
	Wfov = float(args.Wfov)
	Hpatch = float(args.Hpatch)
	Wpatch = float(args.Wpatch)
	Zpatch = float(args.Zpatch)
	out_base = args.out_base
	
	### Print feedback
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('                                   getHistoDensity.py                               ')
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('')
	print('')
	print('* List of CSV files to process: {}'.format(inlist))
	print('* Source histological image height: {} um'.format(Hfov))
	print('* Source histological image width: {} um'.format(Wfov))
	print('* Target patch height: {} um'.format(Hpatch))
	print('* Target patch width: {} um'.format(Wpatch))
	print('* Target slice thickness: {} um'.format(Zpatch))
	print('* Output NIFTI files: {}_cellno.nii, {}_cellsmm2.nii, {}_cellareaum2.nii, {}_cellfract.nii'.format(args.out_base,args.out_base,args.out_base,args.out_base))
	print('* Output binary files: {}_cellno.npy, {}_cellsmm2.npy, {}_cellareaum2.npy, {}_cellfract.npy'.format(args.out_base,args.out_base,args.out_base,args.out_base))
	print('')
	print('')
	
	### Run code
	run(inlist,Hfov,Wfov,Hpatch,Wpatch,Zpatch,out_base)
		
	### Done
	print('')
	print('     ... Done')
	print('')

