### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

import argparse, os, sys
import nibabel as nib
import numpy as np
import pickle as pk
from scipy.optimize import minimize
from scipy.special import erf
import multiprocessing as mp
import time



def run(mrifile, mriseq, output):
	''' Get spherical mean signals for a PGSE acquisition 
	    
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
	    (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain).'''

	
	
	### Load MRI measurements and check for consistency
	timeinitial = time.time()
	print('')
	print('    ... loading MRI measurements')
	try:
		m_obj = nib.load(mrifile)
	except:
		print('')
		raise RuntimeError('ERROR: the 4D input file {} does not exist or is not in NIFTI format.'.format(mrifile))
	m_data = m_obj.get_fdata()
	m_size = m_data.shape
	m_size = np.array(m_size)
	m_header = m_obj.header
	m_affine = m_header.get_best_affine()
	m_dims = m_obj.shape
	if m_size.size!=4:
		print('')
		raise RuntimeError('ERROR: the 4D input file {} is not a 4D NIFTI.'.format(mrifile))			 

	### Load MRI sequence parameters and check for consistency
	print('')
	print('    ... loading MRI sequence information')
	acqpar = np.loadtxt(mriseq)
	totvols = acqpar.shape[1]   # Number of MRI volumes
	if(m_size[3]!=totvols):
		print('')
		raise RuntimeError('ERROR: the number of measurements in {} and {} do not match'.format(mrifile,mriseq))
	if(acqpar.shape[0]!=3):
		print('')
		raise RuntimeError('ERROR: the sequence parameter file {} must have 3 rows'.format(mriseq))

	
	#### Compute spherical mean signals at fixed b, grad. duration delta and grad. separation Delta
	print('')
	print('    ... computing spherical mean signals')
	bvlist = acqpar[0,:]
	acqpar[1,bvlist==0] = 0.0
	acqpar[2,bvlist==0] = 0.0
	bv_unique = np.unique(bvlist)
	is_first_sphmean = True
	for bb in range(0,bv_unique.size):
		acqpar_selbvals = acqpar[:,bvlist==bv_unique[bb]]
		sub_bval = m_data[:,:,:,bvlist==bv_unique[bb]]
		gdur_selbvals = acqpar_selbvals[1,:]
		gdur_selbvals_unique = np.unique(gdur_selbvals)
		for dd in range(0,gdur_selbvals_unique.size):
			acqpar_selbvals_selgdur = acqpar_selbvals[:,gdur_selbvals==gdur_selbvals_unique[dd]]
			sub_bval_gdur = sub_bval[:,:,:,gdur_selbvals==gdur_selbvals_unique[dd]]
			gsep_selbvals_selgdur = acqpar_selbvals_selgdur[2,:]
			gsep_selbvals_selgdur_unique = np.unique(gsep_selbvals_selgdur) 
			for qq in range(0,gsep_selbvals_selgdur_unique.size):
				
				# Get all volumes with fixed b-value, gradient duration and gradient separation
				acqpar_selbvals_selgdur_selgsep = acqpar_selbvals_selgdur[:,gsep_selbvals_selgdur==gsep_selbvals_selgdur_unique[qq]]
				sub_bval_gdur_gsep = sub_bval_gdur[:,:,:,gsep_selbvals_selgdur==gsep_selbvals_selgdur_unique[qq]]

				# Calculate spherical mean
				buff_sphmean_unbias = np.nanmean(sub_bval_gdur_gsep,axis=3)
				
				# Add a singleton dimension so that different spherical means can be stacked
				sub_bval_gdur_gsep_sphmean_unbias = np.zeros((buff_sphmean_unbias.shape[0],buff_sphmean_unbias.shape[1],buff_sphmean_unbias.shape[2],1))
				sub_bval_gdur_gsep_sphmean_unbias[:,:,:,0] = buff_sphmean_unbias
				del buff_sphmean_unbias

				# Stack spherical mean to previously computed spherical mean along 4th dimension
				if(is_first_sphmean):
					print('           --- spherical mean for (b,delta,Delta) = ({} s/mm2, {} ms, {} ms)'.format(acqpar_selbvals_selgdur_selgsep[0,0],acqpar_selbvals_selgdur_selgsep[1,0],acqpar_selbvals_selgdur_selgsep[2,0]))
					sphmean_data = np.copy(sub_bval_gdur_gsep_sphmean_unbias)
					acqpar_sphmean = np.zeros((3,1))
					acqpar_sphmean[0,0] = acqpar_selbvals_selgdur_selgsep[0,0]
					acqpar_sphmean[1,0] = acqpar_selbvals_selgdur_selgsep[1,0]
					acqpar_sphmean[2,0] = acqpar_selbvals_selgdur_selgsep[2,0]
					is_first_sphmean = False

				else:
					print('           --- spherical mean for (b,delta,Delta) = ({} s/mm2, {} ms, {} ms)'.format(acqpar_selbvals_selgdur_selgsep[0,0],acqpar_selbvals_selgdur_selgsep[1,0],acqpar_selbvals_selgdur_selgsep[2,0]))
					sphmean_data = np.concatenate((sphmean_data,sub_bval_gdur_gsep_sphmean_unbias),axis=3)
					acqpar_sphmean_new = np.zeros((3,1))
					acqpar_sphmean_new[0,0] = acqpar_selbvals_selgdur_selgsep[0,0]
					acqpar_sphmean_new[1,0] = acqpar_selbvals_selgdur_selgsep[1,0]
					acqpar_sphmean_new[2,0] = acqpar_selbvals_selgdur_selgsep[2,0]
					acqpar_sphmean = np.concatenate((acqpar_sphmean,acqpar_sphmean_new),axis=1)
	Nsphmean = acqpar_sphmean.shape[1]    # Number of spherical mean measurements			

	## Save Spherical mean signals if required
	print('')
	print('    ... saving spherical mean signals as a 4D NIFTI file')
	buffer_header = m_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not	
	sm_obj = nib.Nifti1Image(sphmean_data,m_obj.affine,buffer_header)
	nib.save(sm_obj, '{}.nii.gz'.format(output))
	np.savetxt('{}.acq.txt'.format(output),acqpar_sphmean,fmt='%.2f', delimiter=' ')

	

	### Done
	timefinal = time.time()
	print('    ... done - it took {} sec'.format(timefinal - timeinitial))
	print('')




# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Compute spherical mean signal for a PGSE acquisition. Third-party dependencies: nibabel, numpy, scipy. Last successful test with nibabel=5.1.0, scipy=1.10.1, numpy=1.23.5. Francesco Grussu, Vall d Hebron Institute of Oncology <fgrussu@vhio.net>. Copyright (c) 2024-2025, Vall d Hebron Institute of Oncology (VHIO), Barcelona, Spain. All rights reserved.')
	parser.add_argument('dwi', help='path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values and, potentially, multiple diffusion times')
	parser.add_argument('scheme', help='path of a text file storing the sequence parameters as a space-separated text file, made of 3 rows x M columns, where column m = 1, ..., M corresponds to the m-th volume of the input NIFTI file. -- First row: b-values in s/mm2; -- Second row: gradient duration small delta in ms; -- Third row: gradient separation Large Delta in ms')
	parser.add_argument('out', help='root file name of output files. Two output files will be saved: a NIFTI file with the spherical mean signal (*.nii.gz) and a textfile with the corresponding sequence parameters (*acq.txt). Output *nii.gz: file 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation. Output *acq.txt: space-separated text file storing the sequence parameters correspnding to the spherical means. It features the same number of columns as *nii.gz, and has 3 lines: -- First row: b-values in s/mm2 -- Second row: gradient duration small delta in ms -- Third row: gradient separation Large Delta in ms')
	args = parser.parse_args()
  
	### Get input arguments
	sfile = args.dwi
	schfile = args.scheme
	outroot = args.out
	
	### Print feedback
	print('')
	print('***********************************************************************')
	print('                         getSphericalMean.py                           ')
	print('***********************************************************************')
	print('')
	print('** 4D NIFTI file with MRI measurements: {}'.format(sfile))
	print('** MRI sequence parameter text file: {}'.format(schfile))
	print('** Output root name: {}'.format(outroot))

		
	### Run computation
	run(sfile, schfile, outroot) 
	
	### Job done
	sys.exit(0)



