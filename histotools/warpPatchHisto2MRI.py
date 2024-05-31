### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

import argparse, os, sys
import numpy as np
import nibabel as nib
import pickle as pk
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric

class HistoMRIWarp():
	'''HistoMRIWarp is a class to manage 2D histology-to-MRI registration based on
	   DiPy symmetric diffeomorphic registration.
	   Author: Francesco Grussu, Vall d Hebron Institute of Oncology
	           <fgrussu@vhio.net><francegrussu@gmail.com>
	'''		                       
	def __init__(self,niter=100,itol=0.001,stp=0.25,sgm=0.2,otol=1e-05,levpyr=[960,480,240,120,60,30],preproc=None):
		''' Constructor initialising the hyper-parameters for 2D histology-to-MRI registration
		
		obj = warpPatchHisto2MRI.HistoMRIWarp(niter=100,itol=0.001,stp=0.25,sgm=0.2,otol=1e-05,levpyr=[960,480,240,120,60,30],preproc=None)
		
		INPUT ARGUMENTS
		* niter: number of iterations to be performed by the displacement field inversion algorithm  
		* itol: the displacement field inversion algorithm will stop iterating when the inversion error falls below this threshold
		* stp: length of the maximum displacement vector of the update displacement field at each iteration
		* sgm: parameter of the scale-space smoothing kernel
		* otol: the optimisation will stop when the estimated derivative of the energy profile w.r.t. time falls below this threshold
		* levpyr: list storing the number of iterations at each pyramidal layer (None if no pyramidal representation is used)
		* preproc: string storing pre-processing steps to be performed before image registration is performed. Steps can be: LR (left-right), 
		           UD (up-down), LRUD(left-right, followed by up-down; it is the same as up-down followed by left-right), ROT90 (anti-clockwise 
		           rotatation of 90deg), ROT180 (anti-clockwise rotatation of 180deg), ROT270 (anti-clockwise rotatation of 270deg) 
		
		The exact meaning of the input parameters in relation to DiPy registration code is available in the DiPy main documentation:
		https://dipy.org/documentation/1.4.0./reference/dipy.align
		
		RETURNS
		obj: object of HistoMRIWarp class.  
		
		Each object of the HistoMRIWarp class has the following public members:
		* niter: stores the value used for input argument niter
		* itol: stores the value used for input argument itol
		* stp: stores the value used for input argument stp
		* sgm: stores the value used for input argument sgm
		* otol: stores the value used for input argument otol
		* levpyr: stores the list used for input argument levpyr
		* preproc: stores the value used for input argument preproc
		* xfm: stores the calculated warping transformation (initialised to NaN)
		
		Each object of the HistoMRIWarp class has the following methods:
		* run(): to calculate a warping transformation histology-to-MRI and apply it to quantitative histological maps    
		'''
		self.niter=niter
		self.itol=itol
		self.stp=stp
		self.sgm=sgm
		self.otol=otol
		self.levpyr=levpyr
		self.preproc=preproc
		self.xfm = np.nan
		
						
	def run(self, mrifile, mrislice, histofile, outroot, maplist=None):
		''' Warp 2D histology to MRI and optionally apply the warping field to patch-wise histological maps
		
		    HistoMRIWarp.run(mrifile, mrislice, histofile, outroot, maplist=None)
		    
		    INPUT PARAMETERS
		    * mrifile:    NIFTI file storing a binary mask of the tissue on MRI. The tissue must be contained 
		                  in one slice only (i.e. MRI slice from which histological material was derived)
		    * mrislice:   index of the MRI slice to which histology must be registered 
		                  (slices will be evaluated along the third image dimension; set mri_slice to 0 for the 1st slice) 		                  
		    * histofile:  NIFTI file storing a binary mask of the tissue on histology. The NIFTI file must have size 
		                  NxMx1 (i.e. one slice only, stored along the third dimension)
		    * outroot:    root for output file names. The following files will be stored:
		                  - *_histo2mri_tissuemask.nii: histological tissue mask warped to MRI
		                  - *_histo2mri_xfm.bin: pickel binary file storing the histology-to-MRI warping object from class 
		                                         HistoMRIWarp, containing all hyperparameters used to calculate the warping 
		                                         transformation as well as the transformation itself
		                  - *_histo2mri_map1.nii, *_histo2mri_map2.nii, ...: warped parametric maps (if maplist is not None)
		    * maplist:    optional list of patch-wise NIFTI parametric maps from histology to be warped to MRI. 
		                  These should be defined in the same space as histo_mask, and the same warping transformation 
		                  will be applied. The warped maps will be stored in files ending as *map1.nii, *map2.nii, ... etc 
		                  (same order as list in maplist)
		
		'''
		
		## Load MRI 
		mri_obj = nib.load(mrifile)
		mri_data = mri_obj.get_fdata()
		static = mri_data[:,:,mrislice]

		## Load histology
		hist_obj = nib.load(histofile)
		hist_data = hist_obj.get_fdata()
		try:
			moving = hist_data[:,:,0]
		except:
			moving = hist_data
			
		## Apply any preprocessing steps if required
		if self.preproc is not None:
			if self.preproc=='LR':
				moving = np.fliplr(moving)
			elif self.preproc=='UD':
				moving = np.flipud(moving)
			elif self.preproc=='LRUD':
				moving = np.flipud( np.fliplr(moving) )
			elif self.preproc=='ROT90':
				moving = np.rot90( moving )
			elif self.preproc=='ROT180':
				moving = np.rot90( np.rot90( moving ) )
			elif self.preproc=='ROT270':
				moving = np.rot90( np.rot90( np.rot90( moving ) ) )
			else:
				raise RuntimeError('ERROR: {} is an unknown preprocessing. Valid values are LR, UD, LRUD, ROT90, ROT180, ROT270'.format(self.preproc))
		
		## Initialise DiPy co-registration tool
		metric = SSDMetric(2)
		sdr = SymmetricDiffeomorphicRegistration(metric,level_iters=self.levpyr,step_length=self.stp,ss_sigma_factor=self.sgm,opt_tol=self.otol,inv_iter=self.niter,inv_tol=self.itol)

		## Run DiPy co-registration and calculate warping field
		mapping = sdr.optimize(static, moving)
		self.xfm = mapping
		
		## Apply warping field to input tissue mask and to any optional parametric maps that the user may have provided
		warped = mapping.transform(moving)
		if maplist is not None:
			hmap_warped_list = []        # List of warped parametric maps
			
			# Loop through parametric maps
			for nn in range(0,len(maplist)):

				# Load nn-th parametric map
				hmap_obj = nib.load(maplist[nn])
				hmap_data = hmap_obj.get_fdata()
				try:
					hmap = hmap_data[:,:,0]
				except:
					hmap = hmap_data[:,:]
				
				# Apply any preprocessing flips if required
				if self.preproc is not None:
					if self.preproc=='LR':
						hmap = np.fliplr(hmap)
					elif self.preproc=='UD':
						hmap = np.flipud(hmap)
					elif self.preproc=='LRUD':
						hmap = np.flipud( np.fliplr(hmap) )
					elif self.preproc=='ROT90':
						hmap = np.rot90( hmap )
					elif self.preproc=='ROT180':
						hmap = np.rot90( np.rot90( hmap ) )
					elif self.preproc=='ROT270':
						hmap = np.rot90( np.rot90( np.rot90( hmap ) ) )
					else:
						raise RuntimeError('ERROR: {} is an unknown preprocessing. Valid values are LR, UD, LRUD, ROT90, ROT180, ROT270'.format(self.preproc))
				
				# Warp it to MRI and store it in the list of warped parametric maps
				hmap_warped = mapping.transform(hmap)
				hmap_warped_list.append(hmap_warped)


		## Save warped NIFTIs
		# Header information from MRI
		buffer_header = mri_obj.header
		buffer_header.set_data_dtype('float64')   # Make sure we save outputs as a float64
		buffer_affine = mri_obj.affine
		
		# Save warped tissue mask map
		buffer_data = np.zeros(mri_data.shape)
		buffer_data[:,:,mrislice] = warped
		out_obj = nib.Nifti1Image(buffer_data,buffer_affine,buffer_header)
		nib.save(out_obj, '{}_histo2mri_tissuemask.nii'.format(outroot))

		# Save warped parametric maps, if provided
		if maplist is not None:
			# Loop through parametric maps
			for nn in range(0,len(maplist)):
			        
			        # Save the nn-th parametric map
				buffer_data = np.zeros(mri_data.shape)
				buffer_data[:,:,mrislice] = hmap_warped_list[nn]
				out_obj = nib.Nifti1Image(buffer_data,buffer_affine,buffer_header)
				nib.save(out_obj, '{}_histo2mri_map{}.nii'.format(outroot,nn+1))
		
		# Save transformation
		xfm_file = open('{}_histo2mri_xfm.bin'.format(outroot),'wb')
		pk.dump(self,xfm_file,pk.HIGHEST_PROTOCOL)
		xfm_file.close()	
			
			


if __name__ == "__main__":

	
	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Co-register patch-wise 2D histology information stored in NIFTI format to a co-localised NIFTI MRI scan with DiPy symmetric diffeomorphic registration')
	parser.add_argument('mri_mask', help='NIFTI file storing a binary mask of the tissue on MRI. The tissue must be contained in one slice only (i.e. MRI slice from which histological material was derived), and slices will be evaluated along the third image dimension')
	parser.add_argument('mri_slice', help='index of the MRI slice to which histology must be registered (slices will be evaluated along the third image dimension; set mri_slice to 0 for the 1st slice)')
	parser.add_argument('histo_mask', help='NIFTI file storing a binary mask of the tissue on histology. The NIFTI file must have size NxMx1 (i.e. one slice only, stored along the third dimension)')
	parser.add_argument('out_root', help='Root for output file names. The following files will be stored: *_histo2mri_tissuemask.nii: histological tissue mask warped to MRI; *_histo2mri_xfm.bin: pickel binary file storing the histology-to-MRI warping object from class HistoMRIWarp, containing all hyperparameters used to calculate the warping transformation as well as the transformation itself;  *_histo2mri_map1.nii, *_histo2mri_map2.nii, ...: warped parametric maps (if flag --histo_maps is used)')
	parser.add_argument('--histo_maps', metavar='<map1,map2,...>', help='optional list of patch-wise NIFTI parametric maps from histology to be warped to MRI. These should be defined in the same space as histo_mask, and the same warping transformation will be applied. Files should be separated by commas (e.g. celldensity.nii,vasculardensity.nii or celldensity.nii,vasculardensity.nii,proliferation.nii etc). The warped maps will be stored in files ending as *map1.nii, *map2.nii, ... etc (same order as --histo_maps input)')
	parser.add_argument('--preproc', metavar='<list>', help='preprocessing steps to be performed before image registration is performed. Steps can be: LR (left-right), UD (up-down), LRUD (left-right, followed by up-down; it is the same as up-down followed by left-right), ROT90 (anti-clockwise rotatation of 90deg), ROT180 (anti-clockwise rotatation of 180deg), ROT270 (anti-clockwise rotatation of 270deg). Default: none')
	parser.add_argument('--leviter', metavar='<list>', default='960,480,240,120,60,30', help='the number of iterations at each level of the Gaussian Pyramid (the length of the list defines the number of pyramid levels to be used) (description from https://dipy.org/documentation/1.4.0./reference/dipy.align/). Default: 960,480,240,120,60,30. In case of multiple pyramid levels, separate numbers by commas (e.g., 128,64,32 or 48,24)')
	parser.add_argument('--inviter', metavar='<value>', default='100',  help='number of iterations to be performed by the displacement field inversion algorithm (description from https://dipy.org/documentation/1.4.0./reference/dipy.align/). Default: 100')
	parser.add_argument('--invtol', metavar='<value>', default='0.001',  help='the displacement field inversion algorithm will stop iterating when the inversion error falls below this threshold (description from https://dipy.org/documentation/1.4.0./reference/dipy.align/). Default: 0.001')
	parser.add_argument('--steplen', metavar='<value>', default='0.25',  help='length of the maximum displacement vector of the update displacement field at each iteration (description from https://dipy.org/documentation/1.4.0./reference/dipy.align/). Default: 0.25')
	parser.add_argument('--sgmfact', metavar='<value>', default='0.2',  help='parameter of the scale-space smoothing kernel. For example, the std. dev. of the kernel will be factor*(2^i) in the isotropic case where i = 0, 1, …, n_scales is the scale (description from https://dipy.org/documentation/1.4.0./reference/dipy.align/). Default: 0.2')
	parser.add_argument('--opttol', metavar='<value>', default='1e-05',  help='the optimisation will stop when the estimated derivative of the energy profile w.r.t. time falls below this threshold (description from https://dipy.org/documentation/1.4.0./reference/dipy.align/). Default: 1e-05')
	args = parser.parse_args()

	### Get input information
	mri_nii = args.mri_mask
	histo_nii = args.histo_mask
	out = args.out_root
	zslice = int(args.mri_slice)
	preprocstr = args.preproc
	inviter = int(args.inviter)
	invtol = float(args.invtol)
	steplen = float(args.steplen)
	sgmfact = float(args.sgmfact)
	opttol = float(args.opttol)
	leviter = args.leviter
	if args.leviter is not None:
		leviter = leviter.split(',')
		leviter = list(map( int,leviter )) 
	if args.histo_maps is not None:
		histo_maps = (args.histo_maps).split(',')
	else:
		histo_maps = None
		
	### Print feedback
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('                               warpPatchHisto2MRI.py                                ')
	print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('')
	print('')
	print('* File storing MRI tissue mask: {}'.format(mri_nii))
	print('* MRI slice of interest: {}'.format(zslice))
	print('* File storing histology tissue mask: {}'.format(histo_nii))
	if args.histo_maps is not None:
		print('* Histological maps: {}'.format(histo_maps))
		
	if args.preproc is not None:
		print('* Preprocessing step: {}'.format(preprocstr))
	else:
		print('* Preprocessing step: none')

	if args.leviter is not None:
		print('* Iterations at each pyramidal level: {}'.format(leviter))
	else:
		print('* No pyramidal encoding')

	print('* Iterations: {}'.format(inviter))
	print('* Target inversion error: {}'.format(invtol))
	print('* Maximum step length at each iteration: {}'.format(steplen))
	print('* Smoothing parameter: {}'.format(sgmfact))
	print('* Tolerance in energy calculation: {}'.format(opttol))
	print('* Output files: {}_histo2mri_tissuemask.nii (warped histological tissue mask), {}_histo2mri_xfm.bin (transformation)'.format(out,out))
	if args.histo_maps is not None:
		print('                (plus warped histological maps {}_histo2mri_map1.nii, ...)'.format(out))
	print('')	
	
	## Print feedback
	print('')
	print('    ... running co-registration')
	print('')
	
	### Create warping object
	mywarp = HistoMRIWarp(niter=inviter,itol=invtol,stp=steplen,sgm=sgmfact,otol=opttol,levpyr=leviter,preproc=preprocstr)
	
	### Run registration
	mywarp.run(mri_nii, zslice, histo_nii, out, maplist=histo_maps)
	
	### Done
	print('')
	print('    ... done')
	print('')

	
