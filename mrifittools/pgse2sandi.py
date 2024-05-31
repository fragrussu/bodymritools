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

def _get_roots():
	# First ten roots of x*J'3/2(x) - 0.5*J3/2(x) = 0 where J3/2(x) is the Bessel function of first kind and order 3/2, while J'3/2 its first-order derivative
	rootvals = np.array([2.081576180816,5.940370139404,9.205839932058,12.404445244044,15.579236315792,18.742645627426,21.899696378997,25.052825050528,28.203361242034,31.352091673521])	
	return rootvals



def _adcsphereGPA(mripar,tpar):

	# Get input stuff
	b_smm2 = mripar[0,:]           # b-value in s/mm2
	gd_ms_array = mripar[1,:]      # gradient duration delta in ms
	gs_ms_array = mripar[2,:]      # gradient separation Delta in ms
	b_msum2 = b_smm2/1000.0        # b-value in ms/um2	
	R_um = tpar[0]                 # Cell radius R in um
	D0_um2ms = tpar[1]             # Intrinsic cytosol diffusivity D0 in um2/ms
	nmeas = mripar.shape[1]        # Number of measurements 

	# Get intra-cellular apparent diffusion coefficient ADCintra in um2/ms
	jroots = _get_roots()                # Roots of x*J'3/2(x) - 0.5*J3/2(x) = 0 where J3/2(x) is the Bessel function of first kind and order 3/2, while J'3/2 its derivative
	Nseries = jroots.size                # Number of terms to be kept in the truncated series
	ADCi_um2ms_array = np.zeros(nmeas)   # Allocate output array for ADCin
	for nn in range(0,nmeas):
		wfactor_um2xms = 0.0         # Weighting factor in um2*ms2
		gd_ms = gd_ms_array[nn]      # Gradient duration delta in ms of current measurement
		gs_ms = gs_ms_array[nn]      # Gradient separation Delta in ms of current measurement
		bv = b_msum2[nn]             # b-value in ms/um2 of current measurement
		if( (gd_ms==0) or (gs_ms==0) or (bv==0) ):
			ADCi_um2ms_array[nn] = np.nan
		else:
			for mm in range(0,Nseries):
				a_um = jroots[mm]/R_um      # alpham expressed in 1/um
				sumval_um2xms =  ( 1.0 / ( D0_um2ms*a_um*a_um*a_um*a_um*(a_um*a_um*R_um*R_um - 2.0) ) ) * ( 2.0*gd_ms - (2.0 + np.exp(-a_um*a_um*D0_um2ms*(gs_ms - gd_ms)) -2.0*np.exp(-a_um*a_um*D0_um2ms*gd_ms) -2.0*np.exp(-a_um*a_um*D0_um2ms*gs_ms) + np.exp(-a_um*a_um*D0_um2ms*(gs_ms + gd_ms)) )/(a_um*a_um*D0_um2ms)  )
				wfactor_um2xms = wfactor_um2xms + sumval_um2xms		
			ADCi_um2ms_array[nn] = ( 2.0/( gd_ms*gd_ms*( gs_ms - gd_ms/3.0 )  ) )*wfactor_um2xms     # ADC in um2/ms
		
	# Return intra-cellular ADC in um2/ms
	return ADCi_um2ms_array



def _sigsphereGPA(mripar,tpar):

	# Get input stuff
	b_smm2 = mripar[0,:]           # b-value in s/mm2
	gd_ms_array = mripar[1,:]      # gradient duration delta in ms
	gs_ms_array = mripar[2,:]      # gradient separation Delta in ms
	b_msum2 = b_smm2/1000.0        # b-value in ms/um2
		
	# Get ADCintra in um2/ms
	adcval = _adcsphereGPA(mripar,tpar)   # tpar = [R,D0] with R -> sphere radius R in um, D0 -> intrinsic sphere diffusivity D0 in um2/ms
	
	# Get MRI signals
	sig = np.exp( - b_msum2*adcval )
	
	# Make sure signal = 1.0 when no diffusion gradients have been turned on
	sig[b_msum2==0] = 1.0
	sig[gs_ms_array==0] = 1.0
	sig[gd_ms_array==0] = 1.0
	
	# Return signals
	return sig



def _sig(mripar,tispar):


	# Get tissue parameters
	din = tispar[0]      # intra-neurite intrinsic diffusivity [um2/ms]
	ds = tispar[1]       # soma intrinsic diffusivity [um2/ms]
	adcec = tispar[2]    # extra-cellular apparent diffusion cofficient [um2/ms] 
	rs = tispar[3]       # soma radius [um]
	fin = tispar[4]      # intra-neurite signal fraction
	fec = tispar[5]      # extra-cellular signal fraction
	s0 = tispar[6]       # Non-DW signal
	
	# Get sequence parameters
	bvals = mripar[0,:]/1000.0     # b-value in ms/um2
	
	# Get characteristic signals for each compartment
	Sec = np.exp(-bvals*adcec)
	Sin = np.sqrt(np.pi/(4.0*bvals*din))*erf( np.sqrt(bvals*din) )
	Sin[bvals==0.0] = 1.0
	Ssoma = _sigsphereGPA(mripar,np.array([rs,ds]))

	# Total signal as in Palombo et al, NeuroImage 2020
	mrisig = s0*fec*Sec + s0*(1-fec)*(fin*Sin + (1-fin)*Ssoma) 
	
	# Return MRI signals
	return mrisig



def _fobj(tisp,meas,acqp,sgm,lnorm,lam,pmin,pmax):

	# Synthesise MRI signal
	measpred = _sig(acqp,tisp)
	
	# Calculate data fidelity term of the objective function (MSE with offset noise floor model)
	fdata = np.nanmean( ( meas - np.sqrt(measpred**2 + sgm**2) )**2 )

	# Calculate regularisation term of the objective function (L1 or L2 norm)
	freg = 0.0
	npar = tisp.size
	if(lnorm==1):
		for pp in range(0,npar):
			freg = freg + np.abs( 0.5 - (tisp[pp] - pmin[pp])/(pmax[pp] - pmin[pp]) )
	elif(lnorm==2):
		for pp in range(0,npar):
			freg = freg + ( 0.5 - (tisp[pp] - pmin[pp])/(pmax[pp] - pmin[pp]) )*( 0.5 - (tisp[pp] - pmin[pp])/(pmax[pp] - pmin[pp]) )
	else:
		raise RuntimeError('ERROR. An L{}-norm is not supported for objective function regularisation'.format(lnorm))
	freg = freg/float(npar)

	# Calculate total objective function
	fout = (1.0 - lam)*fdata + lam*freg

	# Return objective function
	return fout
	


def _procslice(inlist):
	
	# Input information for current MRI slice
	m_data = inlist[0]
	k_data = inlist[1]
	sgm_exist = inlist[2]
	dict_sigs = inlist[3]
	dict_pars = inlist[4]
	prngmin = inlist[5]
	prngmax = inlist[6]
	mriprot = inlist[7]
	sgm_data = inlist[8]
	sl_idx = inlist[9]
	fitalgo = inlist[10]
	Ln = inlist[11]
	Lr = inlist[12]

	# Allocate output maps 
	Nii = m_data.shape[0]
	Njj = m_data.shape[1]
	Din_map_slice = np.zeros((Nii,Njj))
	Ds_map_slice = np.zeros((Nii,Njj))
	Dex_map_slice = np.zeros((Nii,Njj))
	Rs_map_slice = np.zeros((Nii,Njj))
	Fin_map_slice = np.zeros((Nii,Njj))
	Fec_map_slice = np.zeros((Nii,Njj))
	S0_map_slice = np.zeros((Nii,Njj))
	Exitmap_slice = np.zeros((Nii,Njj))
	Fobj_slice = np.zeros((Nii,Njj))

	# Compute L-norm of tissue parameters contained in the grid search dictionary
	Ngrid = dict_pars.shape[0]
	Ntissue = dict_pars.shape[1]
	dict_pars_Lnorm = np.zeros(Ngrid)
	for aa in range(0,Ngrid):
		norm_acc = 0.0
		for bb in range(0,Ntissue):
			if(Ln==1):
				norm_acc = norm_acc + np.abs( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )
			elif(Ln==2):
				norm_acc = norm_acc + ( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )*( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )
			else:
				raise RuntimeError('ERROR. An L{}-norm is not supported for objective function regularisation'.format(Ln))
		dict_pars_Lnorm[aa] = norm_acc/float(Ntissue)

	# Loop through voxels
	for ii in range(0,Nii):
		for jj in range(0,Njj):
				
			## Process voxels within binary mask (if provided)
			if(k_data[ii,jj]==1):
				
				t1111 = time.time()
				# Initialise exit code to 1 (success)
				Exflag = 1
				
				# Get actual MRI measurements
				mvox_original = m_data[ii,jj,:]						
				voxmax = np.max(mvox_original)
				mvox = mvox_original/voxmax
					
				# Deal with excessively high noise levels, if a noise map was provided
				if(sgm_exist):
					# Get noise standard deviation
					sgm_vox_original = sgm_data[ii,jj]
					sgm_vox = sgm_vox_original/voxmax
				else:
					sgm_vox = 0.0
										
					
				# Prepare synthetic signals for dictionary fitting (grid search)
				Nmicro = dict_pars.shape[0]            # Number of combinations of tissue parameters in the grid search
				mvox_mat = np.tile(mvox,(Nmicro,1))    # Matrix of actual MRI measurements
					
				# Find objective function values for grid search dictionary
				fobj_fidelity = (1.0 - Lr)*np.nanmean( (mvox_mat - np.sqrt(dict_sigs**2 + sgm_vox**2))**2 , axis=1) 
				fobj_regulariser = Lr*dict_pars_Lnorm
				fobj_array = fobj_fidelity + fobj_regulariser

				# Estimate tissue parameters via NLLS fitting with regularisation
				try:

					# Get best candidate microstructure (grid search)
					min_idx = np.argmin(fobj_array)
					
					# Get corresponding microstructural parameters
					Din_est = dict_pars[min_idx,0]
					Ds_est = dict_pars[min_idx,1]
					Dex_est = dict_pars[min_idx,2]
					Rs_est = dict_pars[min_idx,3]
					Fin_est = dict_pars[min_idx,4]
					Fec_est = dict_pars[min_idx,5]
					S0_est = dict_pars[min_idx,6]
					
					# Perform non-linear fitting
					param_bound = ((prngmin[0],prngmax[0]),(prngmin[1],prngmax[1]),(prngmin[2],prngmax[2]),(prngmin[3],prngmax[3]),(prngmin[4],prngmax[4]),(prngmin[5],prngmax[5]),(prngmin[6],prngmax[6]),)
					param_init = np.array([Din_est,Ds_est,Dex_est,Rs_est,Fin_est,Fec_est,S0_est])
					modelfit = minimize(_fobj, param_init, method=fitalgo, args=tuple([mvox,mriprot,sgm_vox,Ln,Lr,prngmin,prngmax]), bounds=param_bound)
					param_fit = modelfit.x
					Fobj = modelfit.fun
					Din_est = param_fit[0]
					Ds_est = param_fit[1]
					Dex_est = param_fit[2]
					Rs_est = param_fit[3]
					Fin_est = param_fit[4]
					Fec_est = param_fit[5]
					S0_est = param_fit[6]

					# Check values make sense
					if( np.isnan(Din_est) or np.isnan(Dex_est) or np.isnan(Ds_est) or np.isnan(Rs_est) or np.isnan(Fin_est) or np.isnan(Fec_est) or np.isnan(S0_est) ):
						Din_est = np.nan
						Ds_est = np.nan
						Dex_est = np.nan
						Rs_est = np.nan
						Fin_est = np.nan
						Fec_est = np.nan
						S0_est = np.nan
						Fobj = np.nan
						Exflag = -1
					
				except:
					Din_est = np.nan
					Ds_est = np.nan
					Dex_est = np.nan
					Rs_est = np.nan
					Fin_est = np.nan
					Fec_est = np.nan
					S0_est = np.nan
					Fobj = np.nan
					Exflag = -1					

				# Store microstructural parameters for output
				Din_map_slice[ii,jj] = Din_est
				Ds_map_slice[ii,jj] = Ds_est
				Dex_map_slice[ii,jj] = Dex_est
				Rs_map_slice[ii,jj] = Rs_est
				Fin_map_slice[ii,jj] = Fin_est
				Fec_map_slice[ii,jj] = Fec_est
				S0_map_slice[ii,jj] = S0_est*voxmax
				Exitmap_slice[ii,jj] = Exflag
				Fobj_slice[ii,jj] = Fobj

	# Prepare output list and return it
	outlist = [Din_map_slice,Ds_map_slice,Dex_map_slice,Rs_map_slice,Fin_map_slice,Fec_map_slice,S0_map_slice,Exitmap_slice,Fobj_slice,sl_idx]
	return outlist
	

def run(mrifile, mriseq, output, maskfile=None, noisefile=None, Nword=5, regnorm=2, regw=0.001, pmin=[0.5,2.99,0.5,1.0,0.01,0.01,0.95], pmax=[3.0,3.0,3.0,15.0,0.99,0.99,1.05], nthread=1, slicedim=2, fitting='trust-constr', saveSM=False):
	''' Regularised non-linear least square fitting of the SANDI model from Palombo et al, NeuroImage 2020, doi: 10.1016/j.neuroimage.2020.116835 
	    
	    USAGE
	    run(mrifile, mriseq, output, maskfile=None, noisefile=None, Nword=5, ...
		    pmin=[0.5,2.99,0.5,1.0,0.01,0.01,0.95], pmax=[3.0,3.0,3.0,15.0,0.99,0.99,1.05], nthread=1, slicedim=2, fitting='trust-constr', saveSM=False)
	    
	    * mrifile:    path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values and/or diffusion times 	                  
	                  
	    * mriseq:     path of a text file storing the sequence parameters as a space-separated text file, made of 3 rows x M columns, 
					  where column m = 1, ..., M corresponds to the m-th volume of the input NIFTI file. 
					  -- First row: b-values in s/mm2 
					  -- Second row: gradient duration small delta in ms
					  -- Third row: gradient separation Large Delta in ms
	                  
	    * output:     root file name of output files; output NIFTIs will be stored as double-precision floating point images 
	                  (FLOAT64), and the file names will end in:
	                  *_Dinum2ms.nii (intrinsic intra-neurite diffusivity in um2/ms),
	                  *_Dsomaum2ms.nii (intrinsic soma diffusivity in um2/ms),
	                  *_Dexum2ms.nii (extra-neurite, extra-soma apparent diffusion coefficient in um2/ms),
	                  *_Rsomaum.nii (soma radius in um),
	                  *_Fin.nii (intra-neurite signal fraction),
	                  *_Fex.nii (extra-neurite, extra-soma signal fraction), 
	                  *_S0.nii (estimate of the MRI signal at b = 0), 
	                  *_exit.nii (voxel-wise exit code; -1: warning, error in model fitting; 0: background; 
	                              1 successful parameter estimation).
	                  
	    * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
	                  
	    * noisefile:  3D noise standard deviation map in NIFTI format (used noise floor modelling) 
	                  
	    * Nword:      number of values to test for each tissue parameter grid search (default: 5)

	    * regnorm:    type of L-norm for regularisation (1 for LASSO, 2 for Tikhonov; default: 2) 
		
	    * regw:       weight of the regulariser (ranging in [0.0,1.0]; default: 0.001; set it to 0.0 for 
	                  standard non-linear least square fitting with no regularisation)
	    
	    * pmin:       list of lower bounds for tissue parameters, in this order:
	                  intrinsic intra-neurite diffusivity (um2/ms)
	                  intrinsic soma diffusivity (um2/ms)
	                  extra-neurite, extra-soma apparent diffusion coefficient (um2/ms)
	                  soma radius (um)
	                  intra-neurite signal fraction
	                  extra-neurite, extra-soma signal fraction 
	                  S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)) 
	                  
	                  Default values are [0.5,0.5,0.5,1.0,0.01,0.01,0.95]
	                  If regularisation is used, avoid bounds that are exactly equal to 0.0

	    * pmax:       list of upper bounds for tissue parameters, in this order:
	                  intrinsic intra-neurite diffusivity (um2/ms)
	                  intrinsic soma diffusivity (um2/ms)
	                  extra-neurite, extra-soma apparent diffusion coefficient (um2/ms)
	                  soma radius (um)
	                  intra-neurite signal fraction
	                  extra-neurite, extra-soma signal fraction 
	                  S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)) 
	                  
	                  Default values are [3.0,3.0,3.0,12.0,0.99,0.99,1.05]
	                  If regularisation is used, avoid bounds that are exactly equal to 0.0
	    
	    * nthread:    number of threads to be used for computation (default: 1, single thread)
	    
	    * slicedim:   image dimension along which parallel computation will be exectued when nthread > 1
	                  (can be 0, 1, 2 -- default slicedim=2, implying parallel processing along 3rd 
	                  image dimension)

	    * fitting:   algorithm to be used for constrained objective function minimisation in non-linear fitting. 
	                  Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" 
	                  (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)

	    * saveSM:    boolean flag indicating whether the spherical mean signals should be saved or not. Defalut = False. If True,
	                  If True, the following two output files will be produced:
	                  *_SphMean.nii: 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation
	                  *_SphMean.acq.txt: space-separated text file storing the sequence parameters correspnding to *_SphMean.nii. It features
	                                     the same number of columns as *_SphMean.nii, and has 3 lines:
	                                     -- First row: b-values in s/mm2 
	                                     -- Second row: gradient duration small delta in ms
	                                     -- Third row: gradient separation Large Delta in ms
	    	    
	    Third-party dependencies: nibabel, numpy, scipy
	    Last successful test with nibabel=5.1.0, scipy=1.10.1, numpy=1.23.5
	    
	    Author: Francesco Grussu, University College London
		       <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>
		       	    
	    Code released under BSD Two-Clause license
	    Copyright (c) 2020-2023 University College London
	    All rights reserved'''


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
	acqpar = np.loadtxt(mriseq)
	totvols = acqpar.shape[1]   # Number of MRI volumes
	if(m_size[3]!=totvols):
		print('')
		raise RuntimeError('ERROR: the number of measurements in {} and {} do not match'.format(mrifile,mriseq))
	if(acqpar.shape[0]!=3):
		print('')
		raise RuntimeError('ERROR: the sequence parameter file {} must have 3 rows'.format(mriseq))

	#### Check consistency of some fitting parameters
	regnorm = int(regnorm)
	Nword = int(Nword)
	regw = float(regw)
	if( not( (regnorm==1) or (regnorm==2) ) ):
		raise RuntimeError('ERROR. An L{}-norm is not supported for objective function regularisation'.format(regnorm))
	if( (regw<0.0) or (regw>1.0)):
		raise RuntimeError('ERROR. The regularisation weight was set to {}, but it must be >= 0.0 and <= 1.0'.format(regw))
	if(Nword<0):
		raise RuntimeError('ERROR. The grid search depth for each tissue parameters was set to {} words, but it must be an integer >= 1'.format(Nword))
	
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
	if(saveSM):
		print('')
		print('    ... saving spherical mean signals as a 4D NIFTI file')
		buffer_header = m_obj.header
		buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not	
		sm_obj = nib.Nifti1Image(sphmean_data,m_obj.affine,buffer_header)
		nib.save(sm_obj, '{}_SphMean.nii'.format(output))
		np.savetxt('{}_SphMean.acq.txt'.format(output),acqpar_sphmean,fmt='%.2f', delimiter=' ')

	### Create dictionary of synthetic parameters and corresponding MRI signals
	print('')
	print('    ... creating dictionary of synthetic MRI signals for grid search')
	Din_list = np.linspace(pmin[0],pmax[0],Nword)
	Ds_list = np.linspace(pmin[1],pmax[1],Nword)
	Dex_list = np.linspace(pmin[2],pmax[2],Nword)
	Rs_list = np.linspace(pmin[3],pmax[3],Nword)
	Fin_list = np.linspace(pmin[4],pmax[4],Nword)
	Fex_list = np.linspace(pmin[5],pmax[5],Nword)
	S0_list = np.linspace(pmin[6],pmax[6],Nword)
	Din_array,Ds_array,Dex_array,Rs_array,Fin_array,Fex_array,S0_array = np.meshgrid(Din_list,Ds_list,Dex_list,Rs_list,Fin_list,Fex_list,S0_list)
	Din_array = Din_array.flatten()
	Ds_array = Ds_array.flatten()
	Dex_array = Dex_array.flatten()
	Rs_array = Rs_array.flatten()
	Fin_array = Fin_array.flatten()
	Fex_array = Fex_array.flatten()
	S0_array = S0_array.flatten()
	Nmicro = Din_array.size
	dict_pars = np.zeros((Nmicro,7))
	dict_pars[:,0] = Din_array
	dict_pars[:,1] = Ds_array 
	dict_pars[:,2] = Dex_array
	dict_pars[:,3] = Rs_array
	dict_pars[:,4] = Fin_array 
	dict_pars[:,5] = Fex_array 
	dict_pars[:,6] = S0_array
	dict_sigs = np.zeros((Nmicro,Nsphmean))
	for qq in range(0,Nmicro):
		din = Din_array[qq]
		ds = Ds_array[qq] 
		dex = Dex_array[qq]
		rs = Rs_array[qq]
		fin = Fin_array[qq]
		fex = Fex_array[qq] 
		s0 = S0_array[qq]
		dict_sigs[qq,:] = _sig(acqpar_sphmean,np.array([din,ds,dex,rs,fin,fex,s0]))
	print('           --- {} synthetic signals generated'.format(Nmicro))
	
	### Allocate output parametric maps
	Dinmap = np.zeros((m_size[0],m_size[1],m_size[2]))     # Allocate output: intrinsic intra-neurite diffusivity in um2/ms
	Dsmap = np.zeros((m_size[0],m_size[1],m_size[2]))      # Allocate output: intrinsic soma diffusivity in um2/ms
	Dexmap = np.zeros((m_size[0],m_size[1],m_size[2]))     # Allocate output: apparent extra-cellular diffusion coefficient in um2/ms
	Rsmap = np.zeros((m_size[0],m_size[1],m_size[2]))      # Allocate output: soma radius in um
	Finmap = np.zeros((m_size[0],m_size[1],m_size[2]))     # Allocate output: intra-neurite signal fraction
	Fexmap = np.zeros((m_size[0],m_size[1],m_size[2]))     # Allocate output: extra-neurite signal fraction 
	S0map = np.zeros((m_size[0],m_size[1],m_size[2]))      # Allocate output: non-DW (b=0) signal level
	Exitmap = np.zeros((m_size[0],m_size[1],m_size[2]))    # Allocate output: exit code map
	Fobjmap = np.zeros((m_size[0],m_size[1],m_size[2]))    # Allolcate output: Fobj map
	
	### Processing
	print('')
	print('    ... processing -- please wait')
	
	# Prepare information for current MRI slice
	inputlist = [] 
	for ww in range(0,m_size[slicedim]):

		if(slicedim==0):
		
			k_data_sl = k_data[ww,:,:]
			m_data_sl = sphmean_data[ww,:,:,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[ww,:,:]
			else:
				sgm_data_sl = None	
		
		elif(slicedim==1):
		
			k_data_sl = k_data[:,ww,:]
			m_data_sl = sphmean_data[:,ww,:,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[:,ww,:]
			else:
				sgm_data_sl = None			
		
		elif(slicedim==2):
		
			k_data_sl = k_data[:,:,ww]
			m_data_sl = sphmean_data[:,:,ww,:]
			if(sgm_exist):
				sgm_data_sl = sgm_data[:,:,ww]
			else:
				sgm_data_sl = None		
			
		else:
			raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 

			
		# Input information for current MRI slice		
		sliceinfo = [m_data_sl,k_data_sl,sgm_exist,dict_sigs,dict_pars,pmin,pmax,acqpar_sphmean,sgm_data_sl,ww,fitting,regnorm,regw]
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
			Dinmap_sl = myresults[0] 
			Dsmap_sl = myresults[1]
			Dexmap_sl = myresults[2]
			Rsmap_sl = myresults[3]
			Finmap_sl = myresults[4]
			Fecmap_sl = myresults[5]
			S0map_sl = myresults[6]
			Exitmap_sl = myresults[7]
			Fobj_sl = myresults[8]
			myslice = myresults[9]
			
			if(slicedim==0):
				Dinmap[myslice,:,:] = Dinmap_sl
				Dsmap[myslice,:,:] = Dsmap_sl
				Dexmap[myslice,:,:] = Dexmap_sl
				Rsmap[myslice,:,:] = Rsmap_sl
				Finmap[myslice,:,:] = Finmap_sl
				Fexmap[myslice,:,:] = Fecmap_sl
				S0map[myslice,:,:] = S0map_sl
				Exitmap[myslice,:,:] = Exitmap_sl
				Fobjmap[myslice,:,:] = Fobj_sl
				
			elif(slicedim==1):
				Dinmap[:,myslice,:] = Dinmap_sl
				Dsmap[:,myslice,:] = Dsmap_sl
				Dexmap[:,myslice,:] = Dexmap_sl
				Rsmap[:,myslice,:] = Rsmap_sl
				Finmap[:,myslice,:] = Finmap_sl
				Fexmap[:,myslice,:] = Fecmap_sl
				S0map[:,myslice,:] = S0map_sl
				Exitmap[:,myslice,:] = Exitmap_sl
				Fobjmap[:,myslice,:] = Fobj_sl	
		
			elif(slicedim==2):
				Dinmap[:,:,myslice] = Dinmap_sl
				Dsmap[:,:,myslice] = Dsmap_sl
				Dexmap[:,:,myslice] = Dexmap_sl
				Rsmap[:,:,myslice] = Rsmap_sl
				Finmap[:,:,myslice] = Finmap_sl
				Fexmap[:,:,myslice] = Fecmap_sl
				S0map[:,:,myslice] = S0map_sl
				Exitmap[:,:,myslice] = Exitmap_sl
				Fobjmap[:,:,myslice] = Fobj_sl	
							
			else:
				raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 		
			
		
	
	# Single CPU at work	
	else:
	
		for ww in range(0, m_size[slicedim]):
			myresults = _procslice(inputlist[ww]) 
			Dinmap_sl = myresults[0] 
			Dsmap_sl = myresults[1]
			Dexmap_sl = myresults[2]
			Rsmap_sl = myresults[3]
			Finmap_sl = myresults[4]
			Fecmap_sl = myresults[5]
			S0map_sl = myresults[6]
			Exitmap_sl = myresults[7]
			Fobj_sl = myresults[8]
			myslice = myresults[9]
			
			if(slicedim==0):
				Dinmap[ww,:,:] = Dinmap_sl
				Dsmap[ww,:,:] = Dsmap_sl
				Dexmap[ww,:,:] = Dexmap_sl
				Rsmap[ww,:,:] = Rsmap_sl
				Finmap[ww,:,:] = Finmap_sl
				Fexmap[ww,:,:] = Fecmap_sl
				S0map[ww,:,:] = S0map_sl
				Exitmap[ww,:,:] = Exitmap_sl
				Fobjmap[ww,:,:] = Fobj_sl
				
			elif(slicedim==1):
				Dinmap[:,ww,:] = Dinmap_sl
				Dsmap[:,ww,:] = Dsmap_sl
				Dexmap[:,ww,:] = Dexmap_sl
				Rsmap[:,ww,:] = Rsmap_sl
				Finmap[:,ww,:] = Finmap_sl
				Fexmap[:,ww,:] = Fecmap_sl
				S0map[:,ww,:] = S0map_sl
				Exitmap[:,ww,:] = Exitmap_sl
				Fobjmap[:,ww,:] = Fobj_sl
		
			elif(slicedim==2):
				Dinmap[:,:,ww] = Dinmap_sl
				Dsmap[:,:,ww] = Dsmap_sl
				Dexmap[:,:,ww] = Dexmap_sl
				Rsmap[:,:,ww] = Rsmap_sl
				Finmap[:,:,ww] = Finmap_sl
				Fexmap[:,:,ww] = Fecmap_sl
				S0map[:,:,ww] = S0map_sl
				Exitmap[:,:,ww] = Exitmap_sl
				Fobjmap[:,:,ww] = Fobj_sl
							
			else:
				raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 	
		
		
	### Save output NIFTIs				
	print('')
	print('    ... saving output files')
	buffer_header = m_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
	
	din_obj = nib.Nifti1Image(Dinmap,m_obj.affine,buffer_header)
	nib.save(din_obj, '{}_Dinum2ms.nii'.format(output))

	ds_obj = nib.Nifti1Image(Dsmap,m_obj.affine,buffer_header)
	nib.save(ds_obj, '{}_Dsomaum2ms.nii'.format(output))

	dex_obj = nib.Nifti1Image(Dexmap,m_obj.affine,buffer_header)
	nib.save(dex_obj, '{}_Dexum2ms.nii'.format(output))
	
	rs_obj = nib.Nifti1Image(Rsmap,m_obj.affine,buffer_header)
	nib.save(rs_obj, '{}_Rsomaum.nii'.format(output))

	fin_obj = nib.Nifti1Image(Finmap,m_obj.affine,buffer_header)
	nib.save(fin_obj, '{}_Fin.nii'.format(output))

	fex_obj = nib.Nifti1Image(Fexmap,m_obj.affine,buffer_header)
	nib.save(fex_obj, '{}_Fex.nii'.format(output))
	
	s0_obj = nib.Nifti1Image(S0map,m_obj.affine,buffer_header)
	nib.save(s0_obj, '{}_S0.nii'.format(output))

	exit_obj = nib.Nifti1Image(Exitmap,m_obj.affine,buffer_header)
	nib.save(exit_obj, '{}_exit.nii'.format(output))
	
	fobj_obj = nib.Nifti1Image(Fobjmap,m_obj.affine,buffer_header)
	nib.save(fobj_obj, '{}_fobj.nii'.format(output))

	### Done
	timefinal = time.time()
	print('    ... done - it took {} sec'.format(timefinal - timeinitial))
	print('')




# Run the module as a script when required
if __name__ == "__main__":

	### Print help and parse arguments
	parser = argparse.ArgumentParser(description='Regularised non-linear least square fitting of the SANDI model from Palombo et al, NeuroImage 2020, doi: 10.1016/j.neuroimage.2020.116835. Third-party dependencies: nibabel, numpy, scipy. Last successful test with nibabel=5.1.0, scipy=1.10.1, numpy=1.23.5. Author: Francesco Grussu, University College London <f.grussu@ucl.ac.uk> <francegrussu@gmail.com>. Code released under BSD Two-Clause license. Copyright (c) 2020-2023 University College London. All rights reserved.')
	parser.add_argument('dwi', help='path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values ')
	parser.add_argument('scheme', help='path of a text file storing the sequence parameters as a space-separated text file, made of 3 rows x M columns, where column m = 1, ..., M corresponds to the m-th volume of the input NIFTI file. -- First row: b-values in s/mm2; -- Second row: gradient duration small delta in ms; -- Third row: gradient separation Large Delta in ms')
	parser.add_argument('out', help='root file name of output files; output NIFTIs will be stored as double-precision floating point images (FLOAT64), and the file names will end in:  *_Dinum2ms.nii (intrinsic intra-neurite diffusivity in um2/ms),  *_Dsomaum2ms.nii (intrinsic soma diffusivity in um2/ms), *_Dexum2ms.nii (extra-neurite, extra-soma apparent diffusion coefficient in um2/ms), *_Rsomaum.nii (soma radius in um), *_Fin.nii (intra-neurite signal fraction), *_Fex.nii (extra-neurite, extra-soma signal fraction), *_S0.nii (estimate of the MRI signal at b = 0),  *_exit.nii (voxel-wise exit code; -1: warning, error in model fitting; 0: background; 1 successful parameter estimation).')
	parser.add_argument('--mask', metavar='<file>', help='3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)')
	parser.add_argument('--noise', metavar='<file>', help='3D noise standard deviation map in NIFTI format (used for noise floor modelling)')
	parser.add_argument('--nw', metavar='<num>', default='5', help='number of values to test for each unknown tissue parameter in the grid search (it must be an integer; default 5)')
	parser.add_argument('--reg', metavar='<Lnorm,weight>', help='comma-separated list of parameters for fitting regularisation specifying i) the type of L-norm (1 for LASSO, 2 for Tikhonov), ii) the weight of the regulariser, ranging in [0.0,1.0]. Default: 2,0.001 (L2 norm, with a weight of 0.001). Set 2,0.0 for a standard non-linear least square fitting with no regularisation.')
	parser.add_argument('--pmin', metavar='<p1min,...,p7min>', help='comma-separated list of lower bounds for tissue parameters, in this order: intrinsic intra-neurite diffusivity (um2/ms), intrinsic soma diffusivity (um2/ms),  extra-neurite, extra-soma apparent diffusion coefficient (um2/ms), soma radius (um), intra-neurite signal fraction, extra-neurite extra-soma signal fraction, and S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)). Defalut are 0.5,2.99,0.5,1.0,0.01,0.01,0.95. If regularisation is used, avoid bounds that are exactly equal to 0.0.')
	parser.add_argument('--pmax', metavar='<p1max,...,p7max>', help='comma-separated list of upper bounds for tissue parameters, in this order: intrinsic intra-neurite diffusivity (um2/ms), intrinsic soma diffusivity (um2/ms),  extra-neurite, extra-soma apparent diffusion coefficient (um2/ms), soma radius (um), intra-neurite signal fraction, extra-neurite extra-soma signal fraction, and S0, described as non-DW signal normalised by the maximum observed signal (S0 = S0true/max(signal)). Defalut are 3.0,3.0,3.0,15.0,0.99,0.99,1.05. If regularisation is used, avoid bounds that are exactly equal to 0.0.')
	parser.add_argument('--ncpu', metavar='<num>', default='1', help='number of threads to be used for computation (default: 1, single thread)')
	parser.add_argument('--sldim', metavar='<num>', default='2', help='image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)')
	parser.add_argument('--nlalgo', metavar='<string>', default='trust-constr', help='algorithm to be used for constrained objective function minimisation in non-linear fitting. Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)')
	parser.add_argument('--sm', metavar='<flag>', default='0', help='flag indicating whether the spherical mean signals should be saved or not. Set to 0 or 1 (defalut = 0, do not save spherical mean signals). If set to 1, the following two output files will be produced: *_SphMean.nii: 4D NIFTI file storing a series of spherical mean signals at fixed b, grad. dur, grad. separation; *_SphMean.acq.txt: space-separated text file storing the sequence parameters correspnding to *_SphMean.nii. It features the same number of columns as *_SphMean.nii, and has 3 lines (first row: b-values in s/mm2; second row: gradient duration small delta in ms; third row: gradient separation Large Delta in ms). Anything different from 0 will be treated as True (save spherical mean).')
	args = parser.parse_args()
  
	### Get input arguments
	sfile = args.dwi
	schfile = args.scheme
	outroot = args.out
	maskfile = args.mask
	sgmfile = args.noise
	nword = int(args.nw)
	cpucnt = int(args.ncpu)
	slptr = int(args.sldim)
	algofit = args.nlalgo
	sm_flag = int(args.sm)
	if(sm_flag==0):
		SaveSMnifti = False
	else:
		SaveSMnifti = True

	# Lower bounds
	if args.pmin is not None:
		pMin = (args.pmin).split(',')
		pMin = np.array( list(map( float, pMin )) )
	else:
		pMin = np.array([0.5,2.99,0.5,1.0,0.01,0.01,0.95])
	
	# Upper bounds
	if args.pmax is not None:
		pMax = (args.pmax).split(',')
		pMax = np.array( list(map( float, pMax )) )
	else:
		pMax = np.array([3.0,3.0,3.0,15.0,0.99,0.99,1.05])
	
	# Regularisation options
	if args.reg is not None:
		repars = (args.reg).split(',')
		repars = np.array( list(map( float, repars )) )
		Lnorm = int(repars[0])
		Lweight = float(repars[1])
	else:
		Lnorm = 2
		Lweight = 0.001
	
	### Print feedback
	print('')
	print('***********************************************************************')
	print('                              pgse2sandi.py                            ')
	print('***********************************************************************')
	print('')
	print('** 4D NIFTI file with MRI measurements: {}'.format(sfile))
	print('** MRI sequence parameter text file: {}'.format(schfile))
	print('** Number of words for each tissue parameter grid search: {}'.format(nword))
	print('** Regularisation options: Lnorm = {}, weight = {}'.format(Lnorm,Lweight))
	print('** Number of threads for parallel slice processing: {}'.format(cpucnt))
	print('** Slice dimension for parallel processing: {}'.format(slptr))
	print('** Lower bound for tissue parameters: {}'.format(pMin))
	print('** Upper bound for tissue parameters: {}'.format(pMax))
	print('** Optimisation algorithm: {}'.format(algofit))
	if( maskfile is not None ):
		print('** Optional binary mask file: {}'.format(maskfile))
	if( sgmfile is not None ):
		print('** Optional 3D NIFTI file with noise standard deviation map: {}'.format(sgmfile))
	print('** Output root name: {}'.format(outroot))
	print('** Save Spherical Mean signals: {}'.format(SaveSMnifti))

		
	### Run computation
	run(sfile, schfile, outroot, maskfile=maskfile, noisefile=sgmfile, Nword=nword, regnorm=Lnorm, regw=Lweight, pmin=pMin, pmax=pMax, nthread=cpucnt, slicedim=slptr, fitting=algofit, saveSM=SaveSMnifti) 
	
	### Job done
	sys.exit(0)



