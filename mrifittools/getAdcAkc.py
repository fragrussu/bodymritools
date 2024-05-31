### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

import argparse, os, sys
import nibabel as nib
import numpy as np
import pickle as pk
from scipy.optimize import minimize
import multiprocessing as mp
import time

def _sig(mybvals,tispar):


	# Get tissue parameters
	adc = tispar[0]
	akc = tispar[1]
	s0 = tispar[2]

	# Convert b-values from s/mm2 to s/um2
	bv = mybvals/1000.0     # b-values in s/um2
	
	# Synthesise MRI signal using the given b-values (b-values are expected in s/um2), ADC and Kurtosis
	mrisig = s0*np.exp(-bv*adc + (akc/6.0)*adc*adc*bv*bv ) 
	
	# Return MRI signals
	return mrisig



def _fobj(tisp,meas,acqp,nf):

	# Synthesise MRI signal
	measpred = _sig(acqp,tisp)
	
	# Calculate objective function value (MSE with offset noise floor model)
	fout = np.nansum( ( meas - np.sqrt(measpred**2 + nf**2) )**2 )

	# Return objective function
	return fout
	


def _procslice(inlist):
	
	# Input information for current MRI slice
	m_data = inlist[0]
	k_data = inlist[1]
	sgm_exist = inlist[2]
	navg = inlist[3]
	dict_sigs = inlist[4]
	dict_pars = inlist[5]
	prngmin = inlist[6]
	prngmax = inlist[7]
	bvals = inlist[8]
	sgm_data = inlist[9]
	sl_idx = inlist[10]
	optalgo = inlist[11]
	
	# Allocate output maps
	Nii = m_data.shape[0]
	Njj = m_data.shape[1]
	ADC_map_slice = np.zeros((Nii,Njj))
	AKC_map_slice = np.zeros((Nii,Njj))
	s0_map_slice = np.zeros((Nii,Njj))
	Exitmap_slice = np.zeros((Nii,Njj))
	Fobj_slice = np.zeros((Nii,Njj))
	LogL_slice = np.zeros((Nii,Njj))
	BIC_slice = np.zeros((Nii,Njj))
	AIC_slice = np.zeros((Nii,Njj))

	# Loop through voxels
	for ii in range(0,Nii):
		for jj in range(0,Njj):
				
			## Process voxels within binary mask (if provided)
			if(k_data[ii,jj]==1):
				
				# Initialise exit code to 1 (success)
				Exflag = 1
				
				# Get actual MRI measurements
				mvox = m_data[ii,jj,:]						
				voxmax = np.nanmax(mvox)
				mvox = mvox/voxmax
					
				# Deal with excessively high noise levels, if a noise map was provided
				if(sgm_exist):
				
					# Get noise standard deviation
					sgm_vox = sgm_data[ii,jj]
					sgm_vox = sgm_vox/voxmax
										
					
				# Find synthetic signals that most resemble the actual MRI measurement: no bootstrapping
				if(True):         # Legacy if statement - apologies for the poor coding practice
				
					# Prepare synthetic signals for dictionary fitting (grid search)
					Nmicro = dict_pars.shape[0]            # Number of combinations of tissue parameters in the grid search
					mvox_mat = np.tile(mvox,(Nmicro,1))    # Matrix of actual MRI measurements
						
					# Check whether noise floor modelling is required
					if(sgm_exist):
						if( ~np.isnan(sgm_vox) and ~np.isinf(sgm_vox) ):
							nfloor = sgm_vox*np.sqrt(0.5*np.pi)*np.sqrt(float(navg))
						else:
							nfloor = 0.0		
						mse_array = np.nansum( ( mvox_mat - np.sqrt(dict_sigs**2 + nfloor**2) )**2 , axis=1)
						Nmri = mvox_mat.shape[1]    # Number of measurements for log-likelihood computation
					else:
						nfloor = 0.0
						mse_array = np.nansum( (mvox_mat - dict_sigs)**2 , axis=1)
					
					# Estimate tissue parameters via dictionary fitting
					try:
						# Get best microstructure
						min_idx = np.argmin(mse_array)
						
						# Get corresponding microstructural parameters
						ADC_est = dict_pars[min_idx,0]
						AKC_est = dict_pars[min_idx,1]
						S0_est = dict_pars[min_idx,2]
						Fobj = np.nanmin(mse_array)   # Objective function at minimum
						
						# Perform non-linear fitting
						param_bound = ((prngmin[0],prngmax[0]),(prngmin[1],prngmax[1]),(prngmin[2],prngmax[2]),)
						param_init = np.array([ADC_est,AKC_est,S0_est])
						modelfit = minimize(_fobj, param_init, method=optalgo, args=tuple([mvox,bvals,nfloor]), bounds=param_bound)
						param_fit = modelfit.x
						Fobj = modelfit.fun
						ADC_est = param_fit[0]
						AKC_est = param_fit[1]
						S0_est = param_fit[2]
						
						# Compute log-likelihood, BIC and AIC if an estimate for the noise standard deviation was provided
						if(sgm_exist):
							Nukn = 4.0
							LogL = (-0.5/(sgm_vox*sgm_vox))*Fobj - 0.5*Nmri*np.log( np.sqrt(2*np.pi*sgm_vox*sgm_vox) )  # Log-lik
							BIC = -2.0*LogL + Nukn*np.log(Nmri)      # Bayesian information criterion
							AIC = -2.0*LogL + 2.0*Nukn               # Akaike information criterion
						else:
							LogL = np.nan
							BIC = np.nan
							AIC = np.nan
						
						# Check values make sense
						if( np.isnan(ADC_est) or np.isnan(AKC_est) or np.isnan(S0_est) ):
							ADC_est = np.nan
							AKC_est = np.nan
							S0_est = np.nan
							Fobj = np.nan
							LogL = np.nan
							BIC = np.nan
							AIC = np.nan
							Exflag = -1
						
					except:
						ADC_est = np.nan
						AKC_est = np.nan
						S0_est = np.nan
						Fobj = np.nan
						LogL = np.nan
						BIC = np.nan
						AIC = np.nan
						Exflag = -1						


				# Store microstructural parameters for output
				ADC_map_slice[ii,jj] = ADC_est
				AKC_map_slice[ii,jj] = AKC_est
				s0_map_slice[ii,jj] = S0_est*voxmax
				Exitmap_slice[ii,jj] = Exflag
				Fobj_slice[ii,jj] = Fobj
				LogL_slice[ii,jj] = LogL
				BIC_slice[ii,jj] = BIC
				AIC_slice[ii,jj] = AIC


	# Prepare output list and return it
	outlist = [ADC_map_slice,AKC_map_slice,s0_map_slice,Exitmap_slice,Fobj_slice,LogL_slice,BIC_slice,AIC_slice,sl_idx]
	return outlist
	



def run(mrifile, mriseq, output, maskfile=None, noisefile=None, navg=1, Nword=10, pmin=[0.1,0.0,0.5], pmax=[3.0,5.0,5.0], nthread=1, slicedim=2, nlinalgo='trust-constr'):
	''' Fit a phenomenological representation of the diffusion MRI signal based on apparent diffusion and excess kurtosis coefficients (ADC, AKC)
	    
	    USAGE
	    run(mrifile, bvals, output, maskfile=None, noisefile=None, navg=1, ...
	        Nword=10, pmin=[0.1,0.0,0.5], pmax=[3.0,5.0,5.0], nthread=1, slicedim=2)
	    
	    * mrifile:    path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values 	                  
	                  
	    * bvals:      path of a text file storing a list of b-values in s/mm2 (space-separated, in FSL format), with each
	                  element of this list indicating the b-value of the corresponding diffusion-weighted volume in the 
	                  input 4D NIFTI file
	                  
	    * output:     root file name of output files; output NIFTIs will be stored as double-precision floating point images 
	                  (FLOAT64), and the file names will end in:
	                  *_ADC.nii (ADC in um2/ms),
	                  *_AKC.nii (excess kurtosis),
	                  *_S0.nii (estimate of the MRI signal at b = 0), 
	                  *_exit.nii (voxel-wise exit code; -1: warning, error in model fitting; 0: background; 
	                              1 successful parameter estimation).
	                  If a noise map was provided through the noisefile input parameter (see below), then additional output 
	                  NIFTI files storing quality of fit metrics are stored, i.e.: 
	                  *_logL.nii (log-likelihood at maximum) 
	                  *_BIC.nii (Bayesian Information Criterion)
	                  *_AIC.nii (Akaike Information Criterion)
	                  
	    * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
	                  
	    * noisefile:  3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the
	                  expected Rician noise floor.
	                  
	    * navg:       number of signal averages used for acquisition, which is used to get a more accurate estimate of the
	                  noise floor given the estimate of the noise standard deviation (default: 1; ignored if noisefile = None)
	                  Note that in some vendors, this parameter is referred to as number of excitations (NEX)
	                  
	    * Nword:      number of values to test for each tissue parameter grid search (default: 10)
	    
	    * pmin:       list of lower bounds for tissue parameters, in this order:
	                  ADC (um2/ms)
	                  excess kurtosis AKC
	                  S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)) 
	                  
	                  default values are [0.1,0.0,0.5]  

	    * pmax:       list of upper bounds for tissue parameters, in this order:
	                  ADC (um2/ms)
	                  excess kurtosis AKC
	                  S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)) 
	                  
	                  default values are [3.0,5.0,5.0]
	    
	    * nthread:    number of threads to be used for computation (default: 1, single thread)
	    
	    * slicedim:   image dimension along which parallel computation will be exectued when nthread > 1
	                  (can be 0, 1, 2 -- default slicedim=2, implying parallel processing along 3rd 
	                  image dimension)
	                  
	    * nlinalgo:   algorithm to be used for constrained objective function minimisation in non-linear fitting. 
	                  Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" 
	                  (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
	    	    
	    Third-party dependencies: nibabel, numpy, scipy

	    Author: Francesco Grussu, Vall d'Hebron Institute of Oncology, August 2022
		    <fgrussu@vhio.net>'''


	### Get time
	timeinitial = time.time()
	
	### Get number of threads
	nthread = int(nthread)
	ncpu = mp.cpu_count()
	if( (ncpu - 1) < nthread):
		nthread = ncpu - 1

	if(nthread<0):
		nthread = ncpu - 1
		print('WARNING: negative number of threads -- using {} instead'.format(nthread))

	## Get slice dimension for parallel processing
	if( not( (slicedim==0) or (slicedim==1) or (slicedim==2) ) ):
		slicedim = 2
		print('WARNING: invalid image dimension for parallel processing -- using dimension={} ({}-th dimension)'.format(slicedim,slicedim+1))		
	
	### Load MRI measurements and check for consistency
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

	### Load mask and check for consistency
	if (maskfile is not None):
		print('')
		print('    ... loading mask')
		try:
			k_obj = nib.load(maskfile)
		except:
			print('')
			raise RuntimeError('ERROR: the 3D input file {} does not exist or is not in NIFTI format.'.format(maskfile))
		k_data = k_obj.get_fdata()
		k_size = k_data.shape
		k_size = np.array(k_size)
		k_header = k_obj.header
		k_affine = k_header.get_best_affine()
		k_dims = k_obj.shape
		if k_size.size!=3:
			print('')
			raise RuntimeError('ERROR: the 3D input file {} is not a 3D NIFTI.'.format(maskfile))
		if ( (np.sum(k_affine==m_affine)!=16) or (k_dims[0]!=m_dims[0]) or (k_dims[1]!=m_dims[1]) or (k_dims[2]!=m_dims[2]) ):
			print('')
			raise RuntimeError('ERROR: the header geometry of {} and {} do not match.'.format(mrifile,maskfile))
		k_data[k_data>0] = 1
		k_data[k_data<=0] = 0	
	else:
		k_data = np.ones((m_size[0],m_size[1],m_size[2]))
	
	### Load noise standard deviation map
	if (noisefile is not None):
		print('')
		print('    ... loading noise map')
		try:
			sgm_obj = nib.load(noisefile)
		except:
			print('')
			raise RuntimeError('ERROR: the 3D input file {} does not exist or is not in NIFTI format.'.format(noisefile))
		sgm_data = sgm_obj.get_fdata()
		sgm_size = sgm_data.shape
		sgm_size = np.array(sgm_size)
		sgm_header = sgm_obj.header
		sgm_affine = sgm_header.get_best_affine()
		sgm_dims = sgm_obj.shape
		if sgm_size.size!=3:
			print('')
			raise RuntimeError('ERROR: the 3D input file {} is not a 3D NIFTI.'.format(noisefile))
		if ( (np.sum(sgm_affine==m_affine)!=16) or (sgm_dims[0]!=m_dims[0]) or (sgm_dims[1]!=m_dims[1]) or (sgm_dims[2]!=m_dims[2]) ):
			print('')
			raise RuntimeError('ERROR: the header geometry of {} and {} do not match.'.format(mrifile,noisefile))
		
		# Flags
		sgm_exist = True
		sgm_exist = True
			
	else:
		# Flags
		sgm_exist = False
		sgm_exist = False
		
		# Empty variable
		sgm_data = None
	
	### Load MRI sequence parameters and check for consistency
	print('')
	print('    ... loading MRI sequence information')
	bvals = np.loadtxt(mriseq)
	if(m_size[3]!=bvals.size):
		print('')
		raise RuntimeError('ERROR: the number of measurements in {} and {} do not match'.format(mrifile,mriseq))

	### Create dictionary of synthetic parameters and corresponding MRI signals
	print('')
	print('    ... creating dictionary of synthetic MRI signals')
	ADC_list = np.linspace(pmin[0],pmax[0],Nword)
	AKC_list = np.linspace(pmin[1],pmax[1],Nword)
	S0_list = np.linspace(pmin[2],pmax[2],Nword)
	ADC_array, AKC_array, S0_array = np.meshgrid(ADC_list,AKC_list,S0_list)
	ADC_array = ADC_array.flatten()
	AKC_array = AKC_array.flatten()
	S0_array = S0_array.flatten()
	Nmicro = ADC_array.size
	dict_pars = np.zeros((Nmicro,4))
	dict_pars[:,0] = ADC_array
	dict_pars[:,1] = AKC_array
	dict_pars[:,3] = S0_array
	dict_sigs = np.zeros((Nmicro,bvals.size))
	for qq in range(0,Nmicro):
		adcv = ADC_array[qq]
		akcv = AKC_array[qq]
		sig0 = S0_array[qq]
		dict_sigs[qq,:] = sig0*np.exp( -(bvals/1000.0)*adcv + (akcv/6.0)*adcv*adcv*(bvals/1000.0)*(bvals/1000.0) )
	print('        ({} synthetic signals generated)'.format(Nmicro))
	
	
	### Allocate output parametric maps
	ADCmap = np.zeros((m_size[0],m_size[1],m_size[2]))       # Allocate output: ADC in um2/ms
	AKCmap = np.zeros((m_size[0],m_size[1],m_size[2]))       # Allocate output: AKC in um2/ms
	S0map = np.zeros((m_size[0],m_size[1],m_size[2]))        # Allocate output: non-DW signal estimate
	Exitmap = np.zeros((m_size[0],m_size[1],m_size[2]))      # Allocate output: exit code map
	Fobjmap = np.zeros((m_size[0],m_size[1],m_size[2]))      # Allolcate output: Fobj map
	LogLmap = np.zeros((m_size[0],m_size[1],m_size[2]))      # Allolcate output: log-likelihood map
	BICmap = np.zeros((m_size[0],m_size[1],m_size[2]))       # Allolcate output: Bayesian Information Criterion map
	AICmap = np.zeros((m_size[0],m_size[1],m_size[2]))       # Allolcate output: Akaike Information Criterion map
	
	### Processing
	print('')
	print('    ... processing -- please wait')
	
	# Prepare information for current MRI slice
	inputlist = [] 
	for ww in range(0,m_size[slicedim]):

		if(slicedim==0):
		
			k_data_sl = k_data[ww,:,:]
			m_data_sl = m_data[ww,:,:,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[ww,:,:]
			else:
				sgm_data_sl = None	
		
		elif(slicedim==1):
		
			k_data_sl = k_data[:,ww,:]
			m_data_sl = m_data[:,ww,:,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[:,ww,:]
			else:
				sgm_data_sl = None			
		
		elif(slicedim==2):
		
			k_data_sl = k_data[:,:,ww]
			m_data_sl = m_data[:,:,ww,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[:,:,ww]
			else:
				sgm_data_sl = None		
			
		else:
			raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 

		
		
		# Input information for current MRI slice		
		sliceinfo = [m_data_sl,k_data_sl,sgm_exist,navg,dict_sigs,dict_pars,pmin,pmax,bvals,sgm_data_sl,ww,nlinalgo]
		inputlist.append(sliceinfo)	
	
	# Send slice to process in parallel if nthread>1	
	if(nthread>1):
		
		# Create the parallel pool and give jobs to the workers
		fitpool = mp.Pool(processes=nthread)  # Create parallel processes
		fitpool_pids_initial = [proc.pid for proc in fitpool._pool]  # Get PIDs
		fitresults = fitpool.map_async(_procslice,inputlist)          # Send off to parallel processing
			
		# Check whether results are ready
		while not fitresults.ready():
			fitpool_pids_new = [proc.pid for proc in fitpool._pool]  # Get PIDs again
			if fitpool_pids_new!=fitpool_pids_initial:
				print('')
				raise RuntimeError('ERROR: some processes died during parallel fitting.') # Error: some processes where killed, stop everything and throw an error 
			
		# Work done: get results
		fitlist = fitresults.get()

		# Collect fitting output and re-assemble MRI slices		
		for ww in range(0, m_size[slicedim]):
			myresults = fitlist[ww]		
			ADCmap_sl = myresults[0]	
			AKCmap_sl = myresults[1]		
			S0map_sl = myresults[2]	
			Exitmap_sl = myresults[3]
			Fobj_sl = myresults[4]
			LogL_sl = myresults[5]
			BIC_sl = myresults[6]
			AIC_sl = myresults[7]	
			myslice = myresults[8]
			
			if(slicedim==0):
				ADCmap[ww,:,:] = ADCmap_sl
				AKCmap[ww,:,:] = AKCmap_sl
				S0map[ww,:,:] = S0map_sl
				Exitmap[ww,:,:] = Exitmap_sl
				Fobjmap[ww,:,:] = Fobj_sl
				LogLmap[ww,:,:] = LogL_sl
				BICmap[ww,:,:] = BIC_sl
				AICmap[ww,:,:] = AIC_sl
				
			elif(slicedim==1):
				ADCmap[:,ww,:] = ADCmap_sl
				AKCmap[:,ww,:] = AKCmap_sl
				S0map[:,ww,:] = S0map_sl
				Exitmap[:,ww,:] = Exitmap_sl
				Fobjmap[:,ww,:] = Fobj_sl
				LogLmap[:,ww,:] = LogL_sl
				BICmap[:,ww,:] = BIC_sl
				AICmap[:,ww,:] = AIC_sl		
		
			elif(slicedim==2):
				ADCmap[:,:,ww] = ADCmap_sl
				AKCmap[:,:,ww] = AKCmap_sl
				S0map[:,:,ww] = S0map_sl
				Exitmap[:,:,ww] = Exitmap_sl
				Fobjmap[:,:,ww] = Fobj_sl
				LogLmap[:,:,ww] = LogL_sl
				BICmap[:,:,ww] = BIC_sl
				AICmap[:,:,ww] = AIC_sl
							
			else:
				raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 		
			
		
	
	# Single CPU at work	
	else:
	
		for ww in range(0, m_size[slicedim]):
			myresults = _procslice(inputlist[ww]) 
			ADCmap_sl = myresults[0]	
			AKCmap_sl = myresults[1]		
			S0map_sl = myresults[2]	
			Exitmap_sl = myresults[3]
			Fobj_sl = myresults[4]
			LogL_sl = myresults[5]
			BIC_sl = myresults[6]
			AIC_sl = myresults[7]	
			myslice = myresults[8]
			
			if(slicedim==0):
				ADCmap[ww,:,:] = ADCmap_sl
				AKCmap[ww,:,:] = AKCmap_sl
				S0map[ww,:,:] = S0map_sl
				Exitmap[ww,:,:] = Exitmap_sl
				Fobjmap[ww,:,:] = Fobj_sl
				LogLmap[ww,:,:] = LogL_sl
				BICmap[ww,:,:] = BIC_sl
				AICmap[ww,:,:] = AIC_sl
				
			elif(slicedim==1):
				ADCmap[:,ww,:] = ADCmap_sl
				AKCmap[:,ww,:] = AKCmap_sl
				S0map[:,ww,:] = S0map_sl
				Exitmap[:,ww,:] = Exitmap_sl
				Fobjmap[:,ww,:] = Fobj_sl
				LogLmap[:,ww,:] = LogL_sl
				BICmap[:,ww,:] = BIC_sl
				AICmap[:,ww,:] = AIC_sl		
		
			elif(slicedim==2):
				ADCmap[:,:,ww] = ADCmap_sl
				AKCmap[:,:,ww] = AKCmap_sl
				S0map[:,:,ww] = S0map_sl
				Exitmap[:,:,ww] = Exitmap_sl
				Fobjmap[:,:,ww] = Fobj_sl
				LogLmap[:,:,ww] = LogL_sl
				BICmap[:,:,ww] = BIC_sl
				AICmap[:,:,ww] = AIC_sl
							
			else:
				raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 	
		
		
	### Save output NIFTIs				
	print('')
	print('    ... saving output files')
	buffer_header = m_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
	
	adcin_obj = nib.Nifti1Image(ADCmap,m_obj.affine,buffer_header)
	nib.save(adcin_obj, '{}_ADC.nii'.format(output))  
	
	adcex_obj = nib.Nifti1Image(AKCmap,m_obj.affine,buffer_header)
	nib.save(adcex_obj, '{}_AKC.nii'.format(output))
	
	s0_obj = nib.Nifti1Image(S0map,m_obj.affine,buffer_header)
	nib.save(s0_obj, '{}_S0.nii'.format(output))

	exit_obj = nib.Nifti1Image(Exitmap,m_obj.affine,buffer_header)
	nib.save(exit_obj, '{}_exit.nii'.format(output))
	
	fobj_obj = nib.Nifti1Image(Fobjmap,m_obj.affine,buffer_header)
	nib.save(fobj_obj, '{}_fobj.nii'.format(output))

	# If a noise map was provided, we can get LogL, BIC and AIC from FObj and store them
	if(sgm_exist):
	
		logL_obj = nib.Nifti1Image(LogLmap,m_obj.affine,buffer_header)
		nib.save(logL_obj, '{}_logL.nii'.format(output))
		
		bic_obj = nib.Nifti1Image(BICmap,m_obj.affine,buffer_header)
		nib.save(bic_obj, '{}_BIC.nii'.format(output))
		
		aic_obj = nib.Nifti1Image(AICmap,m_obj.affine,buffer_header)
		nib.save(aic_obj, '{}_AIC.nii'.format(output))


	### Done
	timefinal = time.time()
	print('    ... done - it took {} sec'.format(timefinal - timeinitial))
	print('')




# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Fit a phenomenological representation of the diffusion MRI signal acquired at varying b-values based on the apparent diffusion coefficient (ADC) and apparent excess kurtosis coefficient (AKC). Third-party dependencies: nibabel, numpy, scipy. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). Email: <fgrussu@vhio.net>.')
	parser.add_argument('s_file', help='path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values ')
	parser.add_argument('bval_file', help='path of a text file storing a list of b-values in s/mm2 (space-separated, in FSL format), with each element of this list indicating the b-value of the corresponding diffusion-weighted volume in the input 4D NIFTI file')
	parser.add_argument('out', help='root file name of output files; output NIFTIs will be stored as double-precision floating point images (FLOAT64), and the file names will end in: *_ADC.nii (ADC map um2/ms), *_AKC.nii (excess kurtosis map), *_S0.nii (estimate of the MRI signal at b = 0), *_exit.nii (voxel-wise exit code; -1: warning, error in model fitting; 0: background; 1 successful parameter estimation). If a noise map was provided through the noisefile input parameter (see below), then additional output NIFTI files storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood at maximum), *_BIC.nii (Bayesian Information Criterion) and *_AIC.nii (Akaike Information Criterion)')
	parser.add_argument('--mask', metavar='<file>', help='3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)')
	parser.add_argument('--noise', metavar='<file>', help='3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the expected Rician noise floor.')
	parser.add_argument('--savg', metavar='<num>', default='1', help='number of signal averages used for MRI data acquisition (default: 1). This parameter is used for the estimation of the noise floor (it is ignored if the option --noise is not used). Note that in some vendors, this parameter is also referred to as number of excitations (NEX).')
	parser.add_argument('--nw', metavar='<num>', default='10', help='number of values to test for each unknown tissue parameter in the grid search (it must be an integer; default 10)')
	parser.add_argument('--pmin', metavar='<list>', help='comma-separaterd list of lower bounds for tissue parameters, in this order: ADC (um2/ms); Kurtosis excess AKC; non-DW signal S0 (espressed in normalised form, i.e. the true non-DW signal divided by the maximum observed signal as S0 = S0true/max(s)). Example: 0.2,0.0,0.9. Default: 0.1,0.0,0.5')
	parser.add_argument('--pmax', metavar='<list>', help='comma-separaterd list of upper bounds for tissue parameters, in this order: ADC (um2/ms); Kurtosis excess AKC; non-DW signal S0 (espressed in normalised form, i.e. the true non-DW signal divided by the maximum observed signal as S0 = S0true/max(s)). Example: 2.4,3.5,3.5. Default: 3.0,5.0,5.0')
	parser.add_argument('--ncpu', metavar='<num>', default='1', help='number of threads to be used for computation (default: 1, single thread)')
	parser.add_argument('--sldim', metavar='<num>', default='2', help='image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)')
	parser.add_argument('--nlalgo', metavar='<string>', default='trust-constr', help='algorithm to be used for constrained objective function minimisation in non-linear fitting. Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)')
	args = parser.parse_args()
  
	### Get input arguments
	sfile = args.s_file
	schfile = args.bval_file
	outroot = args.out
	maskfile = args.mask
	sgmfile = args.noise
	nex = int(args.savg)
	nword = int(args.nw)
	cpucnt = int(args.ncpu)
	slptr = int(args.sldim)
	nlopt = args.nlalgo
	
	# Lower bounds
	if args.pmin is not None:
		pMin = (args.pmin).split(',')
		pMin = np.array( list(map( float, pMin )) )
	else:
		pMin = np.array([0.1,0.0,0.5])	
	
	# Upper bounds
	if args.pmax is not None:
		pMax = (args.pmax).split(',')
		pMax = np.array( list(map( float, pMax )) )
	else:
		pMax = np.array([3.0,5.0,5.0])	
	
	### Print feedback
	print('')
	print('***********************************************************************')
	print('                            getAdcAkc.py                               ')
	print('***********************************************************************')
	print('')
	print('** 4D NIFTI file with MRI measurements: {}'.format(sfile))
	print('** MRI sequence parameter text file: {}'.format(schfile))
	print('** Number of words for each tissue parameter grid search: {}'.format(nword))
	print('** Number of threads for parallel slice processing: {}'.format(cpucnt))
	print('** Slice dimension for parallel processing: {}'.format(slptr))
	print('** Lower bound for tissue parameters: {}'.format(pMin))
	print('** Upper bound for tissue parameters: {}'.format(pMax))
	if( maskfile is not None ):
		print('** Optional binary mask file: {}'.format(maskfile))
	if( sgmfile is not None ):
		print('** Optional 3D NIFTI file with noise standard deviation map: {}'.format(sgmfile))
		print('** Number of signal averages: {}'.format(nex))
	print('** Output root name: {}'.format(outroot))
		

	### Run computation
	run(sfile, schfile, outroot, maskfile=maskfile, noisefile=sgmfile, navg=nex, Nword=nword, pmin=pMin, pmax=pMax, nthread=cpucnt, slicedim=slptr, nlinalgo=nlopt)
	
	### Job done
	sys.exit(0)
	

