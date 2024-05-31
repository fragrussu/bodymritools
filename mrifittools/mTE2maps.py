### Copyright (c) 2024, Fundació Privada Institut d’Investigació Oncològica de Vall d’Hebron (Vall d'Hebron Institute of Oncology, VHIO, Barcelona, Spain). 
#   All rights reserved.
#   This code is distributed under the Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)).
#   Its use MUST also comply with the requirements of the individual licenses of all its dependencies.

import argparse, os, sys
import nibabel as nib
import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
import time

def _sigMonoExp(te,tispar):
	
	# Tissue parameters
	tA = tispar[0]
	s0sig = tispar[1]

	# Synthesise multi-echo magnitude signal decay and return it
	megesig = s0sig*np.exp(-te/tA)
	return megesig



def _sigBiExp(te,tispar):

	# Tissue parameters
	tA = tispar[0]
	fA = tispar[1]
	tB = tispar[2]
	s0sig = tispar[3]

	# Synthesise multi-echo magnitude signal decay and return it
	megesig = s0sig*( fA*np.exp(-te/tA) + (1.0 - fA)*np.exp(-te/tB)  )
	return megesig



def _sigBiExpOffres(te,tispar):

	# Tissue parameters
	tA = tispar[0]
	fA = tispar[1]
	tB = tispar[2]
	deltawB = tispar[3]
	s0sig = tispar[4]

	# Synthesise multi-echo magnitude signal decay and return it
	megesig = np.abs( s0sig*(  fA*np.exp(-te/tA) + (1.0 - fA)*np.exp( -te/tB +1j*te*deltawB )  ) )
	return megesig



def _sigTriExp(te,tispar):

	# Tissue parameters
	tA = tispar[0]
	fA = tispar[1]
	tB = tispar[2]
	fB = tispar[3]
	tC = tispar[4]
	s0sig = tispar[5]

	# Synthesise multi-echo magnitude signal decay and return it
	megesig = s0sig*( fA*np.exp(-te/tA) + (1.0 - fA)*( fB*np.exp(-te/tB)  +  (1-fB)*np.exp(-te/tC) )  )
	return megesig



def _sigTriExpOffres(te,tispar):

	# Tissue parameters
	tA = tispar[0]
	fA = tispar[1]
	tB = tispar[2]
	fB = tispar[3]
	deltawB = tispar[4]
	tC = tispar[5]
	deltawC = tispar[6]
	s0sig = tispar[7]

	# Synthesise multi-echo magnitude signal decay and return it
	megesig = s0sig*( fA*np.exp(-te/tA) + (1.0 - fA)*( fB*np.exp( -te/tB +1j*te*deltawB )  +  (1-fB)*np.exp( -te/tC +1j*te*deltawC) )  )
	megesig = np.abs(megesig)
	return megesig




def _sig(te,tispar,modelid):

	# Get the right MRI signal prediction according to the specified model
	if(modelid=='MonoExp'):
		mrs = _sigMonoExp(te,tispar)
	elif(modelid=='BiExp'):
		mrs = _sigBiExp(te,tispar)
	elif(modelid=='BiExpOffres'):
		mrs = _sigBiExpOffres(te,tispar)
	elif(modelid=='TriExp'):
		mrs = _sigTriExp(te,tispar)
	elif(modelid=='TriExpOffres'):
		mrs = _sigTriExpOffres(te,tispar)
	else:
		raise RuntimeError('ERROR: signal model {} is unknown.'.format(modelid))
	
	# Return predicted MRI signals	
	return mrs


	
def _fobj(tisp,meas,acqp,sgm,navg,modelstr,lnorm,lam,pmin,pmax):

	# Synthesise MRI signal
	measpred = _sig(acqp,tisp,modelstr)
	
	# Get noise floor
	nf = sgm*np.sqrt(0.5*np.pi)*np.sqrt(float(navg))

	# Calculate objective function value (MSE with offset noise floor model)
	fdata = np.nanmean( ( meas - np.sqrt(measpred**2 + nf**2) )**2 )

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
	navg = inlist[0]
	acpmat = inlist[1]
	dict_pars = inlist[2]
	dict_sigs = inlist[3]
	k_data = inlist[4]
	m_data = inlist[5]
	sgm_exist = inlist[6] 
	sgm_data = inlist[7]
	slidx = inlist[8]
	prngmin = inlist[9]
	prngmax = inlist[10]
	sigmodel = inlist[11]
	optalgo = inlist[12]
	Ln = inlist[13]
	Lr = inlist[14]

	# Compute L-norm of tissue parameters contained in the grid search dictionary
	Ngrid = dict_pars.shape[0]
	ntissue = dict_pars.shape[1]
	dict_pars_Lnorm = np.zeros(Ngrid)
	for aa in range(0,Ngrid):
		norm_acc = 0.0
		for bb in range(0,ntissue):
			if(Ln==1):
				norm_acc = norm_acc + np.abs( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )
			elif(Ln==2):
				norm_acc = norm_acc + ( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )*( 0.5 - (dict_pars[aa,bb] - prngmin[bb])/(prngmax[bb] - prngmin[bb]) )
			else:
				raise RuntimeError('ERROR. An L{}-norm is not supported for objective function regularisation'.format(Ln))
		dict_pars_Lnorm[aa] = norm_acc/float(ntissue)
  
	# Allocate output maps
	Nii = m_data.shape[0]
	Njj = m_data.shape[1]
	ntissue = prngmin.size   # Number of tissue parameters to estimate depending on the model
	pars_slice = np.zeros((Nii,Njj,ntissue))
	Exitmap_slice = np.zeros((Nii,Njj))      # Exit code will be 0 in the background, 1 where fitting was successful, -1 where fitting failed
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
				mvox = m_data[ii,jj,:]                # MRI measurements
				mvox_max = np.nanmax(mvox)
				mvox = mvox/mvox_max
										
				# Get noise level if it was provided
				if(sgm_exist):				
					# Get noise standard deviation
					sgm_vox = sgm_data[ii,jj]
					sgm_vox = sgm_vox/mvox_max
				else:
					sgm_vox = 0.0

				# Prepare synthetic signals for dictionary fitting (grid search)
				Nmicro = dict_pars.shape[0]            # Number of combinations of tissue parameters in the grid search
				mvox_mat = np.tile(mvox,(Nmicro,1))    # Matrix of actual MRI measurements	
					
				# Find objective function values for grid search dictionary
				fobj_fidelity = (1.0 - Lr)*np.nanmean( (mvox_mat - np.sqrt(dict_sigs**2 + sgm_vox**2))**2 , axis=1) 
				fobj_regulariser = Lr*dict_pars_Lnorm
				fobj_array = fobj_fidelity + fobj_regulariser

				### Perform model fitting
				try:

					## Grid search: fitting based on a discrete dictionary of signals
					min_idx = np.argmin(fobj_array)
					Fobj_grid = np.min(fobj_array)
						
					# Get microstructural parameters corresponding to the selected signal 
					parest = dict_pars[min_idx,:]

					# Prepare fitting bounds and parameter initialisation for optimisation
					param_bound = []
					for yy in range(0,ntissue):
						param_bound.append((prngmin[yy],prngmax[yy]))
					param_init = np.copy(parest)

					# Perform model fitting
					modelfit = minimize(_fobj, param_init, method=optalgo, args=tuple([mvox,acpmat,sgm_vox,navg,sigmodel,Ln,Lr,prngmin,prngmax]), bounds=param_bound)
					param_fit = modelfit.x
					Fobj = modelfit.fun
					if(Fobj>Fobj_grid):
							Exflag = -1
     
					# Compute log-likelihood, BIC and AIC if an estimate for the noise standard deviation was provided
					if(sgm_exist):
						Nmri = float(mvox.size)
						Nukn = float(ntissue)
						Fobj_noreg = _fobj(np.array(param_fit),mvox,acpmat,sgm_vox,navg,sigmodel,Ln,0.0,prngmin,prngmax)        # Value of obj. function with no regularisation for probabilistic calculations
						LogL = (-0.5/(sgm_vox*sgm_vox))*Fobj_noreg*Nmri - 0.5*Nmri*np.log( np.sqrt(2*np.pi*sgm_vox*sgm_vox) )   # Log-likelihood
						BIC = -2.0*LogL + Nukn*np.log(Nmri)      # Bayesian information criterion
						AIC = -2.0*LogL + 2.0*Nukn               # Akaike information criterion
					else:
						LogL = np.nan
						BIC = np.nan
						AIC = np.nan

					# Check if values make sense - check if any NaNs were obtained in the output parameter maps
					if( np.isnan(np.sum(param_fit))  ):
						param_fit = np.nan*np.zeros(ntissue)
						Fobj = np.nan
						LogL = np.nan
						BIC = np.nan
						AIC = np.nan
						Exflag = -1

				except:
					param_fit = np.nan*np.zeros(ntissue)
					Fobj = np.nan
					LogL = np.nan
					BIC = np.nan
					AIC = np.nan
					Exflag = -1

				# Store microstructural parameters for output
				for yy in range(0,ntissue):
					pars_slice[ii,jj,yy] = param_fit[yy]
				Exitmap_slice[ii,jj] = Exflag
				Fobj_slice[ii,jj] = Fobj
				LogL_slice[ii,jj] = LogL
				BIC_slice[ii,jj] = BIC
				AIC_slice[ii,jj] = AIC

	# Prepare output list and return it
	outlist = [pars_slice,Exitmap_slice,Fobj_slice,LogL_slice,BIC_slice,AIC_slice,slidx]
	return outlist




def run(mrifile, tefile, output, maskfile=None, noisefile=None, navg=1, Nword=3, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, nlinalgo='trust-constr', modstr='MonoExp'):
	''' This tool estimates the parameters of a magnitude signal decay model, which is fitted to 
	    multi-echo gradient or spin echo measurements via regularised maximum-likelihood 
	    non-linear least square fitting
	    
	    Third-party dependencies: nibabel, numpy, scipy. 
	    Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3.
	    
	    Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). 
	    Email: <francegrussu@gmail.com> <fgrussu@vhio.net>.
	    
	    USAGE
	    run(mrifile, tefile, output, maskfile=None, noisefile=None, s0file=None, navg=1, ... 
	        Nword=6, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, ...
	        nlinalgo='trust-constr', modstr='MonoExp')
	    
	    * mrifile:    path of a 4D NIFTI file storing M multi-echo gradient or spin echo MRI measurements acquired at varying TE
	                  
	    * tefile:     path of a text file storing a space-separated array of echo times TE in ms, arranged along one row 
	                  
	    * output:     root file name of output files; output NIFTIs will be stored as double-precision floating point images 
	                  (FLOAT64), and the file names will end in 
	                  *_S0.nii (signal level at TE = 0 -- S0 = S(TE=0)),
	                  *_TAms.nii (relaxation time (T2 or T2*) of water pool A, im ms -- TrexA), 
	                  *_fA.nii (total signal fraction of water pool A -- fA),
	                  *_TBms.nii (relaxation time (T2 or T2*) of water pool B, im ms -- TrexB),
	                  *_dOmegaBradms-1.nii (off-resonance angular frequency shift of water pool B with respect to pool A, in rad/ms -- dwB), 
	                  *_fBrel.nii (relative signal fraction of water pool B -- fBrelative; the total signal fraction if fB = fBr x (1 - fA)),
	                  *_TBms.nii (relaxation time (T2 or T2*) of water pool B, im ms -- TrexB),
	                  *_dOmegaBradms-1.nii (off-resonance angular frequency shift of water pool B with respect to pool A, in rad/ms -- dwB),
	                  *_TCms.nii (relaxation time (T2 or T2*) of water pool C, im ms -- TrexC),
	                  *_dOmegaCradms-1.nii (off-resonance angular frequency shift of water pool C with respect to pool A, in rad/ms -- dwC), 
	                  *_exit.nii (voxel-wise exit code; -1: warning, failure in non-linear fitting; 0: background; 
	                  1 successful parameter estimation). 
	                  
	                  If a noise map was provided with the noisefile input parameter, additional output NIFTI files 
	                  storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), 
	                  *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion). 
	                  
	                  The number of parametric maps finally stored depends on the model specified with input parameter modstr
	                  (see below). These will be:
	                  TrexA, S0 for model "MonoExp" (mono-exponential decay); 
	                  TrexA, fA, TrexB, S0 for model "BiExp" (bi-exponential decay); 
	                  TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres" (bi-exponential decay with off-resonance effects)
	                  TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp" (tri-exponential decay); 
	                  TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres" (tri-exponential decay with off-resonance effects)

	    * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
	                  
	    * noisefile:  3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the
	                  expected Rician noise floor.  
					  	                  
	    * navg:       number of signal averages used for acquisition, which is used to get a more accurate estimate of the
	                  noise floor given the estimate of the noise standard deviation (default: 1; ignored if noisefile = None)
	                  Note that in some vendors, this parameter is referred to as number of excitations (NEX)
	                  
	    * Nword:      number of values to test for each tissue parameter in the grid search (default: 3)
	    
	    * pmin:       list or array storing the lower bounds for tissue parameters. These are: 
	                  TrexA, S0 for model "MonoExp"; 
	                  TrexA, fA, TrexB, S0 for model "BiExp"; 
	                  TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres";
	                  TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp"; 
	                  TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres". 
	                  
	                  The symbols stand for: 
	                  -- TrexA -> relaxation time of water pool A (ms)
	                  -- fA -> signal fraction ofwater pool A
	                  -- TrexB -> relaxation time of water pool B (ms)
	                  -- fBrelative -> relative signal fraction of water pool B
	                  -- TrexC -> relaxation time of water pool C (ms)
	                  -- dwB -> off-resonance angular frequency shift of water pool B with respect to A (rad/ms)
	                  -- dwC -> off-resonance angular frequency shift of water pool C with respect to A (rad/ms)
	                  -- S0 --> signal at TE = 0, with respect to the maximum of the measured signal
	                  (parameterisation: S(TE = 0) = S0*max(measured_signal))

	                  Default: 
	                  "2.0,0.8" for parameters TrexA, S0 in model "MonoExp"; 
	                  "2.0,0.01,35.0,0.8" for parameters TrexA, fA, TrexB, S0 in model "BiExp"; 
	                  "2.0,0.01,35.0,-0.4,0.8" for parameters TrexA, fA, TrexB, dwB, S0 in model "BiExpOffres";
	                  "2.0,0.01,15.0,0.01,35.0,0.8" for parameters TrexA, fA, TrexB, fBrelative, TrexC, S0 in model "TriExp"; 
	                  "2.0,0.01,15.0,0.01,-0.4,35.0,-0.4,0.8" for parameters TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 in 
	                  model "TriExpOffres"

	                  For reference, note that the water-fat frequency shit at 3T is equal to 2.8 rad/ms.
	                  Here smaller frequency shifts are to be expected.
	                  
	                  For more information on the models, please look at input parameter modstr below.
	    
	    * pmax:       list or array storing the upper bounds for tissue parameters. These are: 
	                  TrexA, S0 for model "MonoExp"; 
	                  TrexA, fA, TrexB, S0 for model "BiExp"; 
	                  TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres";
	                  TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp"; 
	                  TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres". 
	                  
	                  The symbols stand for: 
	                  -- TrexA -> relaxation time of water pool A (ms)
	                  -- fA -> signal fraction ofwater pool A
	                  -- TrexB -> relaxation time of water pool B (ms)
	                  -- fBrelative -> relative signal fraction of water pool B
	                  -- TrexC -> relaxation time of water pool C (ms)
	                  -- dwB -> off-resonance angular frequency shift of water pool B with respect to A (rad/ms)
	                  -- dwC -> off-resonance angular frequency shift of water pool C with respect to A (rad/ms)
	                  -- S0 --> signal at TE = 0, with respect to the maximum of the measured signal
	                  (parameterisation: S(TE = 0) = S0*max(measured_signal))

	                  Default:
	                  "150.0,50.0" for parameters TrexA, S0 in model "MonoExp"; 
	                  "35.0,0.99,150.0,50.0" for parameters TrexA, fA, TrexB, S0 in model "BiExp"; 
	                  "35.0,0.99,150.0,0.4,50.0" for parameters TrexA, fA, TrexB, dwB, S0 in model "BiExpOffres";
	                  "15.0,0.99,35.0,0.99,150.0,50.0" for parameters TrexA, fA, TrexB, fBrelative, TrexC, S0 in model "TriExp"; 
	                  "15.0,0.99,35.0,0.99,0.4,150.0,0.4,50.0" for parameters TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 in 
	                  model "TriExpOffres"
                   
	                  For reference, note that the water-fat frequency shit at 3T is equal to 2.8 rad/ms.
	                  Here smaller frequency shifts are to be expected.

	                  For more information on the models, please look at input parameter modstr below.           

	    * regnorm:    type of L-norm for regularisation (1 for LASSO, 2 for Tikhonov; default: 2) 
		
	    * regw:       weight of the regulariser (ranging in [0.0,1.0]; default: 0.001; set it to 0.0 for 
	                  standard non-linear least square fitting with no regularisation)  

	    * nthread:    number of threads to be used for computation (default: 1, single thread)
	    
	    * slicedim:   image dimension along which parallel computation will be exectued when nthread > 1
	                  (can be 0, 1, 2 -- default slicedim=2, implying parallel processing along 3rd 
	                  image dimension)
	    
	    * nlinalgo:   algorithm to be used for constrained objective function minimisation in non-linear fitting. 
	                  Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" 
	                  (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)
	    
	    * modstr:     string specifying the signal model to fit. Choose among "MonoExp", "BiExp", "BiExpOffres": 
	                  - "MonoExp" (mono-exponential T2 or T2* decay), 
	                  - "BiExp" (bi-exponential T2 or T2* decay; for T2*, the components are considered to be perfectly in phase), 
	                  - "BiExpOffres" (bi-exponential T2* decay, with one component being off-resonance compared to the other; meaningful only for 
	                     multi-echo gradient echo measurements -- DO NOT USE ON SPIN ECHOES);
	                  - "TriExp" (tri-exponential T2 or T2* decay; for T2*, the components are considered to be perfectly in phase), 
	                  - "TriExpOffres" (tri-exponential T2* decay, with components B and C being off-resonance compared to reference component A; 
					     meaningful only for multi-echo gradient echo measurements -- DO NOT USE ON SPIN ECHOES);
	                  
	                  Default: "MonoExp". 
	                  
	                  Note: models "BiExpOffres" and "TriExpOffres" are practical implementations of Susceptibility Perturbation MRI by 
	                  Santiago I et al, Cancer Research (2019) 79 (9): 2435-2444, doi: 10.1158/0008-5472.CAN-18-3682 
	                  https://doi.org/10.1158/0008-5472.CAN-18-3682
            
	    	    
	    Third-party dependencies: nibabel, numpy, scipy.
	    
	    Developed and validated with versions: 
	    - nibabel 3.2.1
	    - numpy 1.21.5
	    - scipy 1.7.3

	    Author: Francesco Grussu, Vall d'Hebron Institute of Oncology, February 2024
		    <fgrussu@vhio.net> <francegrussu@gmail.com>'''


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
	print('    ... loading echo time (TE) list ')
	telist = np.loadtxt(tefile)
	if(telist.ndim!=1):
		print('')
		raise RuntimeError('ERROR: {} must have size M x 1 and should list TEs as a space-separated raw array'.format(tefile))
	if(m_size[3]!=telist.size):
		print('')
		raise RuntimeError('ERROR: the number of measurements in {} and {} do not match'.format(mrifile,tefile))
	
	## Load tissue parameter lower bounds and assign default values if no bounds were provided  "MonoExp", "BiExp", "BiExpOffres"
	if pmin is not None:
		pmin = np.array(pmin)
		pmin_size = pmin.size
		if(modstr=='MonoExp'):
			if(pmin_size!=2):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 2.'.format(pmin,modstr))
		elif(modstr=='BiExp'):
			if(pmin_size!=4):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 4.'.format(pmin,modstr))
		elif(modstr=='BiExpOffres'):
			if(pmin_size!=5):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 5.'.format(pmin,modstr))
		elif(modstr=='TriExp'):
			if(pmin_size!=6):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 6.'.format(pmin,modstr))
		elif(modstr=='TriExpOffres'):
			if(pmin_size!=8):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 8.'.format(pmin,modstr))
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))		
	else:
		if(modstr=='MonoExp'):
			pmin = np.array([2.0,0.8])
		elif(modstr=='BiExp'):
			pmin = np.array([2.0,0.01,35.0,0.8])
		elif(modstr=='BiExpOffres'):
			pmin = np.array([2.0,0.01,35.0,-0.4,0.8])
		elif(modstr=='TriExp'):
			pmin = np.array([2.0,0.01,15.0,0.01,35.0,0.8])
		elif(modstr=='TriExpOffres'):
			pmin = np.array([2.0,0.01,15.0,0.01,-0.4,35.0,-0.4,0.8])
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
	
	## Load tissue parameter upper bounds and assign default values if no bounds were provided
	if pmax is not None:
		pmax = np.array(pmax)
		pmax_size = pmax.size
		if(modstr=='MonoExp'):
			if(pmax_size!=2):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmax = {} - for model {} it should be 2.'.format(pmax,modstr))
		elif(modstr=='BiExp'):
			if(pmax_size!=4):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmax = {} - for model {} it should be 4.'.format(pmax,modstr))
		elif(modstr=='BiExpOffres'):
			if(pmax_size!=5):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmax = {} - for model {} it should be 5.'.format(pmax,modstr))
		elif(modstr=='TriExp'):
			if(pmax_size!=6):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 6.'.format(pmax,modstr))
		elif(modstr=='TriExpOffres'):
			if(pmax_size!=8):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 8.'.format(pmax,modstr))		
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
	else:
		if(modstr=='MonoExp'):
			pmax = np.array([150.0,50.0])
		elif(modstr=='BiExp'):
			pmax = np.array([35.0,0.99,150.0,50.0])
		elif(modstr=='BiExpOffres'):
			pmax = np.array([35.0,0.99,150.0,0.4,50.0])
		elif(modstr=='TriExp'):
			pmax = np.array([15.0,0.99,35.0,0.99,150.0,50.0])
		elif(modstr=='TriExpOffres'):
			pmax = np.array([15.0,0.99,35.0,0.99,0.4,150.0,0.4,50.0])
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
	
	### Create dictionary of synthetic parameters and corresponding MRI signals
	print('')
	print('    ... creating dictionary of synthetic MRI signals')
	
	# Dictionary of tissue parameters
	if(modstr=='MonoExp'):	
		TA_list = np.linspace(pmin[0],pmax[0],Nword)
		S0_list = np.linspace(pmin[1],pmax[1],Nword)
		TA_array, S0_array = np.meshgrid(TA_list,S0_list)
		TA_array = TA_array.flatten()
		S0_array = S0_array.flatten()
		Nmicro = TA_array.size
		dict_pars = np.zeros((Nmicro,2))
		dict_pars[:,0] = TA_array
		dict_pars[:,1] = S0_array
		
	elif(modstr=='BiExp'):
		TA_list = np.linspace(pmin[0],pmax[0],Nword)
		fA_list = np.linspace(pmin[1],pmax[1],Nword)
		TB_list = np.linspace(pmin[2],pmax[2],Nword)
		S0_list = np.linspace(pmin[3],pmax[3],Nword)
		TA_array, fA_array, TB_array, S0_array = np.meshgrid(TA_list,fA_list,TB_list,S0_list)
		TA_array = TA_array.flatten()
		fA_array = fA_array.flatten()
		TB_array = TB_array.flatten()
		S0_array = S0_array.flatten()
		Nmicro = TA_array.size
		dict_pars = np.zeros((Nmicro,4))
		dict_pars[:,0] = TA_array
		dict_pars[:,1] = fA_array
		dict_pars[:,2] = TB_array
		dict_pars[:,3] = S0_array
	
	elif(modstr=='BiExpOffres'):
		TA_list = np.linspace(pmin[0],pmax[0],Nword)
		fA_list = np.linspace(pmin[1],pmax[1],Nword)
		TB_list = np.linspace(pmin[2],pmax[2],Nword)
		dwB_list = np.linspace(pmin[3],pmax[3],Nword)
		S0_list = np.linspace(pmin[4],pmax[4],Nword)
		TA_array, fA_array, TB_array, dwB_array, S0_array = np.meshgrid(TA_list,fA_list,TB_list,dwB_list,S0_list)
		TA_array = TA_array.flatten()
		fA_array = fA_array.flatten()
		TB_array = TB_array.flatten()
		dwB_array = dwB_array.flatten()
		S0_array = S0_array.flatten()
		Nmicro = TA_array.size
		dict_pars = np.zeros((Nmicro,5))
		dict_pars[:,0] = TA_array
		dict_pars[:,1] = fA_array
		dict_pars[:,2] = TB_array
		dict_pars[:,3] = dwB_array
		dict_pars[:,4] = S0_array

	elif(modstr=='TriExp'):
		TA_list = np.linspace(pmin[0],pmax[0],Nword)
		fA_list = np.linspace(pmin[1],pmax[1],Nword)
		TB_list = np.linspace(pmin[2],pmax[2],Nword)
		fB_list = np.linspace(pmin[3],pmax[3],Nword)
		TC_list = np.linspace(pmin[4],pmax[4],Nword)
		S0_list = np.linspace(pmin[5],pmax[5],Nword)
		TA_array, fA_array, TB_array, fB_array, TC_array, S0_array = np.meshgrid(TA_list,fA_list,TB_list,fB_list,TC_list,S0_list)
		TA_array = TA_array.flatten()
		fA_array = fA_array.flatten()
		fB_array = fB_array.flatten()
		TB_array = TB_array.flatten()
		TC_array = TC_array.flatten()
		S0_array = S0_array.flatten()
		Nmicro = TA_array.size
		dict_pars = np.zeros((Nmicro,6))
		dict_pars[:,0] = TA_array
		dict_pars[:,1] = fA_array
		dict_pars[:,2] = TB_array
		dict_pars[:,3] = fB_array
		dict_pars[:,4] = TC_array
		dict_pars[:,5] = S0_array

	elif(modstr=='TriExpOffres'):
		TA_list = np.linspace(pmin[0],pmax[0],Nword)
		fA_list = np.linspace(pmin[1],pmax[1],Nword)
		TB_list = np.linspace(pmin[2],pmax[2],Nword)
		fB_list = np.linspace(pmin[3],pmax[3],Nword)
		dwB_list = np.linspace(pmin[4],pmax[4],Nword)
		TC_list = np.linspace(pmin[5],pmax[5],Nword)
		dwC_list = np.linspace(pmin[6],pmax[6],Nword)
		S0_list = np.linspace(pmin[7],pmax[7],Nword)
		TA_array, fA_array, TB_array, fB_array, dwB_array, TC_array, dwC_array, S0_array = np.meshgrid(TA_list,fA_list,TB_list,fB_list,dwB_list,TC_list,dwC_list,S0_list)
		TA_array = TA_array.flatten()
		fA_array = fA_array.flatten()
		fB_array = fB_array.flatten()
		TB_array = TB_array.flatten()
		TC_array = TC_array.flatten()
		dwB_array = dwB_array.flatten()
		dwC_array = dwC_array.flatten()
		S0_array = S0_array.flatten()
		Nmicro = TA_array.size
		dict_pars = np.zeros((Nmicro,8))
		dict_pars[:,0] = TA_array
		dict_pars[:,1] = fA_array
		dict_pars[:,2] = TB_array
		dict_pars[:,3] = fB_array
		dict_pars[:,4] = dwB_array
		dict_pars[:,5] = TC_array
		dict_pars[:,6] = dwC_array
		dict_pars[:,7] = S0_array

	else:
		raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
	
	# Dictionary of MRI signals
	npars = dict_pars.shape[1]	
	dict_sigs = np.zeros((Nmicro,telist.size))	
	for qq in range(0,Nmicro):      # Loop through different microstructures in the dictionary
		# Synthesise signals and store them
		synsigs = _sig(telist,dict_pars[qq,:],modstr)
		dict_sigs[qq,:] = synsigs
	print('        ({} synthetic signals generated)'.format(Nmicro))
	
	
	### Allocate output parametric maps
	Tparmap = np.zeros((m_size[0],m_size[1],m_size[2],npars))     # Allocate output: parametric maps to be estimated
	Exitmap = np.zeros((m_size[0],m_size[1],m_size[2]))    # Allocate output:  exit code map
	Fobjmap = np.zeros((m_size[0],m_size[1],m_size[2]))    # Allolcate output: Fobj map
	LogLmap = np.zeros((m_size[0],m_size[1],m_size[2]))     # Allolcate output: log-likelihood map
	BICmap = np.zeros((m_size[0],m_size[1],m_size[2]))      # Allolcate output: Bayesian Information Criterion map
	AICmap = np.zeros((m_size[0],m_size[1],m_size[2]))      # Allolcate output: Akaike Information Criterion map
	
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

		
		sliceinfo = [navg,telist,dict_pars,dict_sigs,k_data_sl,m_data_sl,sgm_exist,sgm_data_sl,ww,pmin,pmax,modstr,nlinalgo,regnorm,regw]   
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

		# Collect fitting output and re-assemble MRI slices	## Recall: the output of _procslice() contains out = [pars_slice,Exitmap_slice,Fobj_slice,LogL_slice,BIC_slice,AIC_slice,slidx]	
		for ww in range(0, m_size[slicedim]):
			myresults = fitlist[ww]		
			allmaps_sl = myresults[0]	
			Exitmap_sl = myresults[1]
			Fobj_sl = myresults[2]
			LogL_sl = myresults[3]
			BIC_sl = myresults[4]
			AIC_sl = myresults[5]	
			myslice = myresults[6]		
			
			if(slicedim==0):
				for uu in range(0,npars):
					Tparmap[myslice,:,:,uu] = allmaps_sl[:,:,uu]
				Exitmap[myslice,:,:] = Exitmap_sl
				Fobjmap[myslice,:,:] = Fobj_sl
				LogLmap[myslice,:,:] = LogL_sl
				BICmap[myslice,:,:] = BIC_sl
				AICmap[myslice,:,:] = AIC_sl
				
			elif(slicedim==1):
				for uu in range(0,npars):
					Tparmap[:,myslice,:,uu] = allmaps_sl[:,:,uu]
				Exitmap[:,myslice,:] = Exitmap_sl
				Fobjmap[:,myslice,:] = Fobj_sl
				LogLmap[:,myslice,:] = LogL_sl
				BICmap[:,myslice,:] = BIC_sl
				AICmap[:,myslice,:] = AIC_sl			
		
			elif(slicedim==2):
				for uu in range(0,npars):
					Tparmap[:,:,myslice,uu] = allmaps_sl[:,:,uu]
				Exitmap[:,:,myslice] = Exitmap_sl
				Fobjmap[:,:,myslice] = Fobj_sl
				LogLmap[:,:,myslice] = LogL_sl
				BICmap[:,:,myslice] = BIC_sl
				AICmap[:,:,myslice] = AIC_sl
							
			else:
				raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 		
			
		
	
	# Single CPU at work	
	else:
	
		for ww in range(0, m_size[slicedim]):
			myresults = _procslice(inputlist[ww]) 
			allmaps_sl = myresults[0]	
			Exitmap_sl = myresults[1]
			Fobj_sl = myresults[2]
			LogL_sl = myresults[3]
			BIC_sl = myresults[4]
			AIC_sl = myresults[5]	
			myslice = myresults[6]	
			
			if(slicedim==0):
				for uu in range(0,npars):
					Tparmap[myslice,:,:,uu] = allmaps_sl[:,:,uu]
				Exitmap[myslice,:,:] = Exitmap_sl
				Fobjmap[myslice,:,:] = Fobj_sl
				LogLmap[myslice,:,:] = LogL_sl
				BICmap[myslice,:,:] = BIC_sl
				AICmap[myslice,:,:] = AIC_sl
				
			elif(slicedim==1):
				for uu in range(0,npars):
					Tparmap[:,myslice,:,uu] = allmaps_sl[:,:,uu]
				Exitmap[:,myslice,:] = Exitmap_sl
				Fobjmap[:,myslice,:] = Fobj_sl
				LogLmap[:,myslice,:] = LogL_sl
				BICmap[:,myslice,:] = BIC_sl
				AICmap[:,myslice,:] = AIC_sl			
		
			elif(slicedim==2):
				for uu in range(0,npars):
					Tparmap[:,:,myslice,uu] = allmaps_sl[:,:,uu]
				Exitmap[:,:,myslice] = Exitmap_sl
				Fobjmap[:,:,myslice] = Fobj_sl
				LogLmap[:,:,myslice] = LogL_sl
				BICmap[:,:,myslice] = BIC_sl
				AICmap[:,:,myslice] = AIC_sl
							
			else:
				raise RuntimeError('ERROR: invalid slice dimension slicedim = {}'.format(slicedim)) 	
		
	
			
	### Save output NIFTIs				
	print('')
	print('    ... saving output files')
	buffer_header = m_obj.header
	buffer_header.set_data_dtype('float64')   # Make sure we save output files float64, even if input is not
	
	# Save exit code, objective function and S0
	exit_obj = nib.Nifti1Image(Exitmap,m_obj.affine,buffer_header)
	nib.save(exit_obj, '{}_exit.nii'.format(output))
	
	fobj_obj = nib.Nifti1Image(Fobjmap,m_obj.affine,buffer_header)
	nib.save(fobj_obj, '{}_fobj.nii'.format(output))

	
	if(modstr=='MonoExp'):
		map_strings = ['TAms.nii','S0.nii']
	elif(modstr=='BiExp'):
		map_strings = ['TAms.nii','fA.nii','TBms.nii','S0.nii']
	elif(modstr=='BiExpOffres'):
		map_strings = ['TAms.nii','fA.nii','TBms.nii','dOmegaBradms-1.nii','S0.nii']
	elif(modstr=='TriExp'):
		map_strings = ['TAms.nii','fA.nii','TBms.nii','fBrel.nii','TCms.nii','S0.nii']
	elif(modstr=='TriExpOffres'):
		map_strings = ['TAms.nii','fA.nii','TBms.nii','fBrel.nii','dOmegaBradms-1.nii','TCms.nii','dOmegaCradms-1.nii','S0.nii']
	else:
		raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
		
	# Save parametric maps
	for uu in range(0,npars):

		# Rescale S0 estimate
		if(map_strings[uu]=='S0.nii'):
			Tparmap[:,:,:,uu] = k_data*Tparmap[:,:,:,uu]*np.nanmax(m_data,axis=3)
			S0fitted = 1.0*Tparmap[:,:,:,uu]
	
		niiout_obj = nib.Nifti1Image(np.squeeze(Tparmap[:,:,:,uu]),m_obj.affine,buffer_header)
		nib.save(niiout_obj, '{}_{}'.format(output,map_strings[uu]))

		
	
	# If a noise map had been provided, we can get LogL, BIC and AIC from FObj and store them
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
	parser = argparse.ArgumentParser(description='This tool estimates the parameters of a magnitude signal decay model, which is fitted to multi-echo gradient or spin echo measurements via regularised maximum-likelihood non-linear least square fitting. Third-party dependencies: nibabel, numpy, scipy. Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). Email: <francegrussu@gmail.com> <fgrussu@vhio.net>.')
	parser.add_argument('s_file', help='path of a 4D NIFTI file storing M multi-echo gradient or spin echo MRI measurements acquired at varying TE')
	parser.add_argument('te_file', help='path of a text file storing a space-separated array of echo times TE in ms, arranged along one row.')
	parser.add_argument('out', help='root file name of output files; output NIFTIs will be stored as double-precision floating point images(FLOAT64), and the file names will end in *_S0.nii (signal level at TE = 0 -- S0 = S(TE=0)), *_TAms.nii (relaxation time (T2 or T2*) of water pool A, im ms -- TrexA), *_fA.nii (total signal fraction of water pool A -- fA), *_TBms.nii (relaxation time (T2 or T2*) of water pool B, im ms -- TrexB), *_dOmegaBradms-1.nii (off-resonance angular frequency shift of water pool B with respect to pool A, in rad/ms -- dwB), *_fBrel.nii (relative signal fraction of water pool B -- fBrelative; the total signal fraction if fB = fBr x (1 - fA)),*_TBms.nii (relaxation time (T2 or T2*) of water pool B, im ms -- TrexB), *_dOmegaBradms-1.nii (off-resonance angular frequency shift of water pool B with respect to pool A, in rad/ms -- dwB),*_TCms.nii (relaxation time (T2 or T2*) of water pool C, im ms -- TrexC), *_dOmegaCradms-1.nii (off-resonance angular frequency shift of water pool C with respect to pool A, in rad/ms -- dwC), *_exit.nii (voxel-wise exit code; -1: warning, failure in non-linear fitting; 0: background; 1 successful parameter estimation). If a noise map was provided with the noisefile input parameter, additional output NIFTI filesstoring quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion). The number of parametric maps finally stored depends on the model specified with input parameter modstr(see below). These will be: TrexA, S0 for model "MonoExp" (mono-exponential decay); TrexA, fA, TrexB, S0 for model "BiExp" (bi-exponential decay); TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres" (bi-exponential decay with off-resonance effects); TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp" (tri-exponential decay); TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres" (tri-exponential decay with off-resonance effects)')
	parser.add_argument('--mask', metavar='<file>', help='3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)')
	parser.add_argument('--noise', metavar='<file>', help='3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the expected Rician noise floor.')
	parser.add_argument('--savg', metavar='<num>', default='1', help='number of signal averages used for MRI data acquisition (default: 1). This parameter is used for the estimation of the noise floor (it is ignored if the option --noise is not used). Note that in some vendors, this parameter is also referred to as number of excitations (NEX).')
	parser.add_argument('--nw', metavar='<num>', default='3', help='number of values to test for each unknown tissue parameter in the grid search (it must be an integer; default 3)')
	parser.add_argument('--pmin', metavar='<list>', help='list storing the lower bounds for tissue parameters. These are: TrexA, S0 for model "MonoExp"; TrexA, fA, TrexB, S0 for model "BiExp"; TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres"; TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp"; TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres".The symbols stand for: -- TrexA -> relaxation time of water pool A (ms) -- fA -> signal fraction ofwater pool A -- TrexB -> relaxation time of water pool B (ms) -- fBrelative -> relative signal fraction of water pool B -- TrexC -> relaxation time of water pool C (ms) -- dwB -> off-resonance angular frequency shift of water pool B with respect to A (rad/ms) -- dwC -> off-resonance angular frequency shift of water pool C with respect to A (rad/ms) -- S0 --> signal at TE = 0, with respect to the maximum of the measured signal(parameterisation: S(TE = 0) = S0*max(measured_signal)). Default:"2.0,0.8" for parameters TrexA, S0 in model "MonoExp"; "2.0,0.01,35.0,0.8" for parameters TrexA, fA, TrexB, S0 in model "BiExp"; "2.0,0.01,35.0,-0.4,0.8" for parameters TrexA, fA, TrexB, dwB, S0 in model "BiExpOffres"; "2.0,0.01,15.0,0.01,35.0,0.8" for parameters TrexA, fA, TrexB, fBrelative, TrexC, S0 in model "TriExp"; "2.0,0.01,15.0,0.01,-0.4,35.0,-0.4,0.8" for parameters TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 inmodel "TriExpOffres". For reference, note that the water-fat frequency shit at 3T is equal to 2.8 rad/ms, and here smaller frequency shifts are to be expected. For more information on the models, please look at option modstr below.')
	parser.add_argument('--pmax', metavar='<list>', help='list or array storing the upper bounds for tissue parameters. These are: TrexA, S0 for model "MonoExp"; TrexA, fA, TrexB, S0 for model "BiExp"; TrexA, fA, TrexB, dwB, S0 for model "BiExpOffres"; TrexA, fA, TrexB, fBrelative, TrexC, S0 for model "TriExp"; TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 for model "TriExpOffres". The symbols stand for: -- TrexA -> relaxation time of water pool A (ms) -- fA -> signal fraction ofwater pool A -- TrexB -> relaxation time of water pool B (ms) -- fBrelative -> relative signal fraction of water pool B -- TrexC -> relaxation time of water pool C (ms) -- dwB -> off-resonance angular frequency shift of water pool B with respect to A (rad/ms) -- dwC -> off-resonance angular frequency shift of water pool C with respect to A (rad/ms) -- S0 --> signal at TE = 0, with respect to the maximum of the measured signal(parameterisation: S(TE = 0) = S0*max(measured_signal)). Default:"150.0,50.0" for parameters TrexA, S0 in model "MonoExp"; "35.0,0.99,150.0,50.0" for parameters TrexA, fA, TrexB, S0 in model "BiExp"; "35.0,0.99,150.0,0.4,50.0" for parameters TrexA, fA, TrexB, dwB, S0 in model "BiExpOffres"; "15.0,0.99,35.0,0.99,150.0,50.0" for parameters TrexA, fA, TrexB, fBrelative, TrexC, S0 in model "TriExp"; "15.0,0.99,35.0,0.99,0.4,150.0,0.4,50.0" for parameters TrexA, fA, TrexB, fBrelative, dwB, TrexC, dwC, S0 inmodel "TriExpOffres". For reference, note that the water-fat angular frequency shit at 3T is equal to 2.8 rad/ms, and here smaller frequency shifts are to be expected. For more information on the models, please look at option modstr below.')
	parser.add_argument('--reg', metavar='<Lnorm,weight>', help='comma-separated list of parameters for fitting regularisation specifying i) the type of L-norm (1 for LASSO, 2 for Tikhonov), ii) the weight of the regulariser, ranging in [0.0,1.0]. Default: 2,0.001 (L2 norm, with a weight of 0.001). Set 2,0.0 for a standard non-linear least square fitting with no regularisation.')
	parser.add_argument('--ncpu', metavar='<num>', default='1', help='number of threads to be used for computation (default: 1, single thread)')
	parser.add_argument('--sldim', metavar='<num>', default='2', help='image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)')
	parser.add_argument('--nlalgo', metavar='<string>', default='trust-constr', help='algorithm to be used for constrained objective function minimisation in non-linear fitting (relevant if --nlfit 1). Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)')
	parser.add_argument('--modstr', metavar='<string>', default='MonoExp', help='string specifying the signal model to fit. Choose among "MonoExp", "BiExp", "BiExpOffres": - "MonoExp" (mono-exponential T2 or T2* decay), - "BiExp" (bi-exponential T2 or T2* decay; for T2*, the components are considered to be perfectly in phase), - "BiExpOffres" (bi-exponential T2* decay, with one component being off-resonance compared to the other; meaningful only for multi-echo gradient echo measurements -- DO NOT USE ON SPIN ECHOES); - "TriExp" (tri-exponential T2 or T2* decay; for T2*, the components are considered to be perfectly in phase), - "TriExpOffres" (tri-exponential T2* decay, with components B and C being off-resonance compared to reference component A; meaningful only for multi-echo gradient echo measurements -- DO NOT USE ON SPIN ECHOES); Default: "MonoExp". Note: models "BiExpOffres" and "TriExpOffres" are practical implementations of Susceptibility Perturbation MRI by Santiago I et al, Cancer Research (2019) 79 (9): 2435-2444, doi: 10.1158/0008-5472.CAN-18-3682 (paper link: https://doi.org/10.1158/0008-5472.CAN-18-3682).')
	args = parser.parse_args()


	### Get input arguments
	sfile = args.s_file
	tefile = args.te_file
	outroot = args.out
	maskfile = args.mask
	sgmfile = args.noise
	nex = int(args.savg)
	nword = int(args.nw)
	cpucnt = int(args.ncpu)
	slptr = int(args.sldim)
	nlopt = args.nlalgo
	mymodel = args.modstr

	### Get maximum TE
	try:
		myTEs = np.loadtxt(tefile)
		myTEmax = np.max(myTEs)
	except:
		raise RuntimeError('ERROR. Something is wrong with echo time file {}'.format(tefile))


	# Lower bounds
	if args.pmin is not None:
		pMin = (args.pmin).split(',')
		pMin = np.array( list(map( float, pMin )) )
	else:
		if(mymodel=='MonoExp'):
			pMin = np.array([2.0,0.8])
		elif(mymodel=='BiExp'):
			pMin = np.array([2.0,0.01,35.0,0.8])
		elif(mymodel=='BiExpOffres'):
			pMin = np.array([2.0,0.01,35.0,-0.4,0.8])
		elif(mymodel=='TriExp'):
			pMin = np.array([2.0,0.01,15.0,0.01,35.0,0.8])
		elif(mymodel=='TriExpOffres'):
			pMin = np.array([2.0,0.01,15.0,0.01,-0.4,35.0,-0.4,0.8])
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(mymodel))
	
	# Upper bounds
	if args.pmax is not None:
		pMax = (args.pmax).split(',')
		pMax = np.array( list(map( float, pMax )) )
	else:
		if(mymodel=='MonoExp'):
			pMax = np.array([150.0,50.0])
		elif(mymodel=='BiExp'):
			pMax = np.array([35.0,0.99,150.0,50.0])
		elif(mymodel=='BiExpOffres'):
			pMax = np.array([35.0,0.99,150.0,0.4,50.0])
		elif(mymodel=='TriExp'):
			pMax = np.array([15.0,0.99,35.0,0.99,150.0,50.0])
		elif(mymodel=='TriExpOffres'):
			pMax = np.array([15.0,0.99,35.0,0.99,0.4,150.0,0.4,50.0])
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(mymodel))
	
	# Regularisation options
	if args.reg is not None:
		fooreg = (args.reg).split(',')
		fooreg = np.array( list(map( float, fooreg )) )
		Lnorm = int(fooreg[0])
		Lweight = float(fooreg[1])
	else:
		Lnorm = 2
		Lweight = 0.001


	### Print feedback
	print('')
	print('***********************************************************************')
	print('                              mTE2maps.py                              ')
	print('***********************************************************************')
	print('')
	print('** 4D NIFTI file with MRI measurements: {}'.format(sfile))
	print('** Echo time list (text file) storing TE in ms: {}'.format(tefile))
	print('** Model to be fitted: {}'.format(mymodel))
	print('** Number of words for each tissue parameter grid search: {}'.format(nword))
	print('** Number of threads for parallel slice processing: {}'.format(cpucnt))
	print('** Slice dimension for parallel processing: {}'.format(slptr))
	print('** Lower bound for tissue parameters: {}'.format(pMin))
	print('** Upper bound for tissue parameters: {}'.format(pMax))
	print('** Fitting regularisation options: Lnorm = {}, weight = {}'.format(Lnorm,Lweight))
	if( maskfile is not None ):
		print('** Optional binary mask file: {}'.format(maskfile))
	if( sgmfile is not None ):
		print('** Optional 3D NIFTI file with noise standard deviation map: {}'.format(sgmfile))
		print('** Number of signal averages: {}'.format(nex))
	print('** Output root name: {}'.format(outroot))
		

	### Run computation
	run(sfile, tefile, outroot, maskfile=maskfile, noisefile=sgmfile, navg=nex, Nword=nword, pmin=pMin, pmax=pMax, regnorm=Lnorm, regw=Lweight, nthread=cpucnt, slicedim=slptr, nlinalgo=nlopt, modstr=mymodel)
	
	### Job done
	sys.exit(0)

