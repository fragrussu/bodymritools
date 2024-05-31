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



def _get_roots():
	# First ten roots of J'1(x) = 0, where J'1(x) is the derivative of the Bessel function of first kind and order 1
	rootvals = np.array([1.84118378, 5.33144277, 8.53631637, 11.7060049, 14.86358863, 18.01552786, 21.16436986, 24.31132686, 27.45705057, 30.60192297])	
	return rootvals



def _adccylperpGPA(mripar,tpar):

	# Get input stuff
	b_smm2 = mripar[0,:]           # b-value in s/mm2
	gd_ms_array = mripar[1,:]      # gradient duration delta in ms
	gs_ms_array = mripar[2,:]      # gradient separation Delta in ms
	b_msum2 = b_smm2/1000.0        # b-value in ms/um2	
	L_um = tpar[0]                 # Cylinder diameter L in um
	R_um = L_um/2.0                # Cylinder radius R = L/2.0 in um
	D0_um2ms = tpar[1]             # Intrinsic cylinder diffusivity D0 in um2/ms
	nmeas = mripar.shape[1]        # Number of measurements 

	# Get intra-axonal apparent diffusion coefficient ADCintra in um2/ms (for large axons)
	jroots = _get_roots()                # Roots of J'1(x) = 0, where J'1(x) is the derivative of the Bessel function of first kind and order 1
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
				num_value =  2.0*D0_um2ms*a_um*a_um*gd_ms - 2.0 + 2.0*np.exp(-D0_um2ms*a_um*a_um*gd_ms) + 2.0*np.exp(-D0_um2ms*a_um*a_um*gs_ms) - np.exp(-D0_um2ms*a_um*a_um*(gs_ms - gd_ms)) - np.exp(-D0_um2ms*a_um*a_um*(gs_ms + gd_ms)) 
				den_value = a_um*a_um*a_um*a_um*a_um*a_um*(R_um*R_um*a_um*a_um - 1.0)
				wfactor_um2xms = wfactor_um2xms + num_value/den_value		
			ADCi_um2ms_array[nn] = ( 2.0/( D0_um2ms*D0_um2ms*gd_ms*gd_ms*( gs_ms - gd_ms/3.0 )  ) )*wfactor_um2xms     # ADC in um2/ms
		
	# Return intra-axonal perpendicular ADC in um2/ms
	# Same as in Van Geldern et al, "Evaluation of restricted diffusion in cylinders. Phosphocreatine in rabbit leg muscle"
	# J Magn Reson B 1994, 103(3):255-60, doi: 10.1006/jmrb.1994.1038
	return ADCi_um2ms_array



def _sigscylperpGPA(mripar,tpar):

	# Get input stuff
	b_smm2 = mripar[0,:]           # b-value in s/mm2
	gd_ms_array = mripar[1,:]      # gradient duration delta in ms
	gs_ms_array = mripar[2,:]      # gradient separation Delta in ms
	b_msum2 = b_smm2/1000.0        # b-value in ms/um2
		
	# Get ADCintra in um2/ms
	adcval = _adccylperpGPA(mripar,tpar)   # tpar = [L,D0] with L -> cell diameter L in um, D0 -> intrinsic cytosol diffusivity D0 in um2/ms
	
	# Get MRI signals
	sig = np.exp( - b_msum2*adcval )
	
	# Make sure signal = 1.0 when no diffusion gradients have been turned on
	sig[b_msum2==0] = 1.0
	sig[gs_ms_array==0] = 1.0
	sig[gd_ms_array==0] = 1.0
	
	# Return signals
	return sig



def _sigDinDexTD(acqpar,tispar):

	# Get sequence parameters
	mybvals = acqpar[0,:]           # bvalues in s/mm2
	ldelta = acqpar[2,:]            # Gradient separation in ms

	# Get tissue parameters
	lval = tispar[0]                # cell size in um2/ms
	d0val = tispar[1]               # intrinsic cell diffusivity in um2/ms
	fval = tispar[2]                # intra-cellular signal fraction
	adcexinf = tispar[3]            # extra-cellular diffusivity in um2/ms
	Betaval = tispar[4]             # time-dependent extra-cellular coefficient in um2
	S0sig = tispar[5]               # non-DW signal level
	
	# Synthesise MRI signal
	dex = adcexinf + Betaval/ldelta
	mrisig = fval*_sigscylperpGPA(acqpar,np.array([lval,d0val])) +  ( 1 - fval )*np.exp(-(mybvals/1000.0)*dex)
	mrisig[mybvals==0] = 1.0
	mrisig = S0sig*mrisig
	
	# Return MRI signals
	return mrisig
	
	
	
def _sigDinDex(acqpar,tispar):

	# Get sequence parameters
	mybvals = acqpar[0,:]           # bvalues in s/mm2

	# Get tissue parameters
	lval = tispar[0]                # cell size in um2/ms
	d0val = tispar[1]               # intrinsic cell diffusivity in um2/ms
	fval = tispar[2]                # intra-cellular signal fraction
	adcexinf = tispar[3]            # extra-cellular diffusivity in um2/ms
	S0sig = tispar[4]               # non-DW signal level
	
	# Synthesise MRI signal
	dex = adcexinf
	mrisig = fval*_sigscylperpGPA(acqpar,np.array([lval,d0val])) +  ( 1 - fval )*np.exp(-(mybvals/1000.0)*dex)
	mrisig[mybvals==0] = 1.0
	mrisig = S0sig*mrisig
	
	# Return MRI signals
	return mrisig
	
	

def _sigDin(acqpar,tispar):

	# Get sequence parameters
	mybvals = acqpar[0,:]           # bvalues in s/mm2

	# Get tissue parameters
	lval = tispar[0]                # cell size in um2/ms
	d0val = tispar[1]               # intrinsic cell diffusivity in um2/ms
	S0sig = tispar[2]               # non-DW signal level
	
	# Synthesise MRI signal
	mrisig = _sigscylperpGPA(acqpar,np.array([lval,d0val]))
	mrisig[mybvals==0] = 1.0
	mrisig = S0sig*mrisig
	
	# Return MRI signals
	return mrisig
	
	

def _sig(acqpar,tispar,modelid):

	# Get the right MRI signal prediction according to the specified model
	if(modelid=='DinDexTD'):
		mrs = _sigDinDexTD(acqpar,tispar)
	elif(modelid=='DinDex'):
		mrs = _sigDinDex(acqpar,tispar)
	elif(modelid=='Din'):
		mrs = _sigDin(acqpar,tispar)
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




def run(mrifile, mriseq, output, maskfile=None, noisefile=None, s0file=None, navg=1, Nword=6, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, nlinalgo='trust-constr', modstr='DinDex'):
	''' This tool estimates the parameters of a 2-compartment model of restricted diffusion within 
	    cylindrical cells (diffusion orthogonal to the cylinder axis) and extra-cellular hindered diffusion, 
	    via regularised non-linear optimisation of a likelihood function under an offset-Gaussian noise model
	    
	    Third-party dependencies: nibabel, numpy, scipy. 
	    Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3.
	    
	    Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). 
	    Email: <francegrussu@gmail.com> <fgrussu@vhio.net>.
	    
	    USAGE
	    run(mrifile, mriseq, output, maskfile=None, noisefile=None, s0file=None, navg=1, ... 
	        Nword=6, pmin=None, pmax=None, regnorm=2, regw=0.001, nthread=1, slicedim=2, ...
	        nlinalgo='trust-constr', modstr='DinDex')
	    
	    * mrifile:    path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values 
	                  and diffusion times
	                  
	    * mriseq:     path of a text file storing information on b-values and diffusion times  corresponding to each 
	                  volume of s_file. The acquisition must be a standard pulsed gradient spin echo
	                  (PGSE, also known as single linear diffusion encoding).
	                  This file must contain a space-separated array of 3 x M elements, arranged along 3 lines 
	                  (first line: b-values in s/mm2; 
	                  second line: gradient duration delta in ms; 
	                  third line: gradient separation Delta in ms).
	                  
	    * output:     root file name of output files; output NIFTIs will be stored as double-precision floating point images 
	                  (FLOAT64), and the file names will end in 
	                  *_Lum.nii (cell diameter L in um), 
	                  *_D0um2ms-1.nii (intrinsic intra-cellular diffusivity D0 in um2/ms), 
	                  *_Dexinfum2ms-1.nii (extra-cellular apparent diffusion coefficient parameter Dexinf in um2/ms), 
	                  *_Betaum2.nii (extra-cellular apparent diffusion coefficient parameter Beta -- note that the 
	                  extra-cellular apparent diffusion coefficient Dex is written as Dex = Dexinf + Beta/t, where t is 
	                  the gradient separation, 
	                  *_fin.nii (intra-cellular tissue signal fraction F),
	                  *_S0.nii (non-DW signal level S0), 
	                  *_cellsmm-2.nii (2D histology-like cellularity C in cells/mm2), 
	                  *_cellsmm-3.nii (cellularity C in cells/mm3), 
	                  *_exit.nii (voxel-wise exit code; -1: warning, failure in non-linear fitting; 0: background; 
	                  1 successful parameter estimation). 
	                  
	                  If a noise map was provided with the noisefile input parameter, additional output NIFTI files 
	                  storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), 
	                  *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion). 
	                  
	                  The number of parametric maps outputted depends on the model specified with input parameter modstr
	                  (see below). These will be: 
	                  L, D0, S0 for model "Din"; 
	                  L, D0, Dexinf, F, S0 for model "DinDex"; 
	                  L, D0, Dexinf, Beta, F, S0 for model "DinDexTD"
	                  
	    * maskfile:   3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)
	                  
	    * noisefile:  3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the
	                  expected Rician noise floor.

	    * s0file:     3D reference non-DW signal S0 in NIFTI format. Relevant only if the fitted model is "Din" (see modstr below). 
		              If provided, S0_Din, i.e., the non-DW signal level estimated via "Din" fitting, will be compared to 
					  S0 to estimate the intra-cellular signal fraction F as F = S0_Din/S0. F will be stored as a NIFTI map
					  as for models "DinDex" and "DinDexTD"			  
					  	                  
	    * navg:       number of signal averages used for acquisition, which is used to get a more accurate estimate of the
	                  noise floor given the estimate of the noise standard deviation (default: 1; ignored if noisefile = None)
	                  Note that in some vendors, this parameter is referred to as number of excitations (NEX)
	                  
	    * Nword:      number of values to test for each tissue parameter in the grid search (default: 6)
	    
	    * pmin:       list or array storing the lower bounds for tissue parameters. These are: 
	                  L,D0,S0 for model "Din"; 
	                  L,D0,F,Dexinf,S0 for model "DinDex"; 
	                  L,D0,F,Dexinf,Beta,S0 for model "DinDexTD". 
	                  
	                  The symbols stand for: 
	                  -- L -> cell size (cell diameter) (um)
	                  -- D0 -> intrinsic intra-cell diffusivity (um2/ms)
	                  -- F -> intra-cellular signal fraction
	                  -- Dexinf -> long-time extra-cellular apparent diffusion coefficient (um2/ms), 
	                  -- Beta -> extra-cellular diffusion-time dependence coefficient (um2) 
	                  Note that the extra-cellular apparent diffusion coeffficient is written as 
	                  Dex = Dexinf + Beta/t, where t is the diffusion time (~ gradient separation) 
	                  -- S0 --> non-DW signal, with respect to the maximum of the measured signal
	                  (parameterisation: S(b = 0) = S0*max(measured_signal))

	                  Default: 
	                  "8.0,0.8,0.01" for model "Din"; 
	                  "8.0,0.8,0.01,0.01,0.6" for model "DinDex"; 
	                  "8.0,0.8,0.01,0.01,0.01,0.6" for model "DinDexTD"
	                  
	                  For more information on the models, please look at input parameter modstr below.
	    
	    * pmax:       list or array storing the upper bounds for tissue parameters. These are: 
	                  L,D0,F,S0 for model "Din"; 
	                  L,D0,F,Dexinf,S0 for model "DinDex"; 
	                  L,D0,F,Dexinf,Beta,S0 for model "DinDexTD". 
	                  
	                  The symbols stand for: 
	                  -- L -> cell size (cell diameter) (um)
	                  -- D0 -> intrinsic intra-cell diffusivity (um2/ms)
	                  -- F -> intra-cellular signal fraction
	                  -- Dexinf -> long-time extra-cellular apparent diffusion coefficient (um2/ms), 
	                  -- Beta -> extra-cellular diffusion-time dependence coefficient (um2) 
	                  Note that the extra-cellular apparent diffusion coeffficient is written as 
	                  Dex = Dexinf + Beta/t, where t is the diffusion time (~ gradient separation)
	                  -- S0 --> non-DW signal, with respect to the maximum of the measured signal
	                  (parameterisation: S(b = 0) = S0*max(measured_signal))

	                  Default: 
	                  "40.0,3.0,1.4" for model "Din"; 
	                  "40.0,3.0,1.0,3.0,1.4" for model "DinDex"; 
	                  "40.0,3.0,1.0,3.0,10.0,1.4" for model "DinDexTD"
	                  
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
	    
	    * modstr:     string specifying the signal model to fit. Choose among: 
	                  - "Din" (extra-vascular signal dominated by intra-cellular diffusion), 
	                  - "DinDex" (extra-vascular signal features both intra-cellular and extra-cellular contributions, without diffusion time 
	                  dependence in the extra-cellular ADC), 
	                  - "DinDexTD" (extra-vascular signal features both intra-cellular and extra-cellular contributions, with diffusion time 
	                  dependence in the extra-cellular ADC).
	                  
	                  Default: "DinDex". 
	                  
	                  Intra-cellular diffusion is modelled using the Gaussian Phase Approximation (GPA) formula for diffusion within sphere 
	                  of Van Geldern et al, "Evaluation of restricted diffusion in cylinders. Phosphocreatine in rabbit leg muscle"
	                  J Magn Reson B 1994, 103(3):255-60, doi: 10.1006/jmrb.1994.1038         
	    	    
	    Third-party dependencies: nibabel, numpy, scipy.
	    
	    Developed and validated with versions: 
	    - nibabel 3.2.1
	    - numpy 1.21.5
	    - scipy 1.7.3

	    Author: Francesco Grussu, Vall d'Hebron Institute of Oncology, November 2022
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
	print('    ... loading MRI sequence information')
	mripar = np.loadtxt(mriseq)
	if(mripar.shape[0]!=3):
		print('')
		raise RuntimeError('ERROR: {} must have size 3 x M and list b-values, grad. dur., grad. sep.'.format(mriseq))
	if(m_size[3]!=mripar.shape[1]):
		print('')
		raise RuntimeError('ERROR: the number of measurements in {} and {} do not match'.format(mrifile,mriseq))

	
	## Load tissue parameter lower bounds and assign default values if no bounds were provided
	if pmin is not None:
		pmin = np.array(pmin)
		pmin_size = pmin.size
		if(modstr=='DinDexTD'):
			if(pmin_size!=6):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 6.'.format(pmin,modstr))
		elif(modstr=='DinDex'):
			if(pmin_size!=5):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 5.'.format(pmin,modstr))
		elif(modstr=='Din'):
			if(pmin_size!=3):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmin = {} - for model {} it should be 3.'.format(pmin,modstr))
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))		
	else:
		if(modstr=='DinDexTD'):
			pmin = np.array([8.0,0.8,0.01,0.01,0.01,0.6])
		elif(modstr=='DinDex'):
			pmin = np.array([8.0,0.8,0.01,0.01,0.6])
		elif(modstr=='Din'):
			pmin = np.array([8.0,0.8,0.01])
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
	
	
	## Load tissue parameter upper bounds and assign default values if no bounds were provided
	if pmax is not None:
		pmax = np.array(pmax)
		pmax_size = pmax.size
		if(modstr=='DinDexTD'):
			if(pmax_size!=6):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmax = {} - for model {} it should be 6.'.format(pmax,modstr))
		elif(modstr=='DinDex'):
			if(pmax_size!=5):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmax = {} - for model {} it should be 5.'.format(pmax,modstr))
		elif(modstr=='Din'):
			if(pmax_size!=3):
				raise RuntimeError('ERROR: wrong number of tissue parameters in lower bounds pmax = {} - for model {} it should be 3.'.format(pmax,modstr))
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
	else:
		if(modstr=='DinDexTD'):
			pmax = np.array([40.0,3.0,1.0,3.0,10.0,1.4])
		elif(modstr=='DinDex'):
			pmax = np.array([40.0,3.0,1.0,3.0,1.4])
		elif(modstr=='Din'):
			pmax = np.array([40.0,3.0,1.4])
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
	
		
	### Create dictionary of synthetic parameters and corresponding MRI signals
	print('')
	print('    ... creating dictionary of synthetic MRI signals')
	
	# Dictionary of tissue parameters
	if(modstr=='DinDexTD'):	
		L_list = np.linspace(pmin[0],pmax[0],Nword)
		Din_list = np.linspace(pmin[1],pmax[1],Nword)
		Fc_list = np.linspace(pmin[2],pmax[2],Nword)
		Dexref_list = np.linspace(pmin[3],pmax[3],Nword)
		Bex_list = np.linspace(pmin[4],pmax[4],Nword)
		S0_list = np.linspace(pmin[5],pmax[5],Nword)
		L_array, Din_array, Fc_array, Dexref_array, Bex_array, S0_array = np.meshgrid(L_list,Din_list,Fc_list,Dexref_list,Bex_list,S0_list)
		L_array = L_array.flatten()
		Din_array = Din_array.flatten()
		Fc_array = Fc_array.flatten()
		Dexref_array = Dexref_array.flatten()
		Bex_array = Bex_array.flatten()
		S0_array = S0_array.flatten()
		Nmicro = L_array.size
		dict_pars = np.zeros((Nmicro,6))
		dict_pars[:,0] = L_array
		dict_pars[:,1] = Din_array
		dict_pars[:,2] = Fc_array
		dict_pars[:,3] = Dexref_array
		dict_pars[:,4] = Bex_array
		dict_pars[:,5] = S0_array
		
	elif(modstr=='DinDex'):
		L_list = np.linspace(pmin[0],pmax[0],Nword)
		Din_list = np.linspace(pmin[1],pmax[1],Nword)
		Fc_list = np.linspace(pmin[2],pmax[2],Nword)
		Dexref_list = np.linspace(pmin[3],pmax[3],Nword)
		S0_list = np.linspace(pmin[4],pmax[4],Nword)
		L_array, Din_array, Fc_array, Dexref_array, S0_array = np.meshgrid(L_list,Din_list,Fc_list,Dexref_list,S0_list)
		L_array = L_array.flatten()
		Din_array = Din_array.flatten()
		Fc_array = Fc_array.flatten()
		Dexref_array = Dexref_array.flatten()
		S0_array = S0_array.flatten()
		Nmicro = L_array.size
		dict_pars = np.zeros((Nmicro,5))
		dict_pars[:,0] = L_array
		dict_pars[:,1] = Din_array
		dict_pars[:,2] = Fc_array
		dict_pars[:,3] = Dexref_array
		dict_pars[:,4] = S0_array
	
	elif(modstr=='Din'):
		L_list = np.linspace(pmin[0],pmax[0],Nword)
		Din_list = np.linspace(pmin[1],pmax[1],Nword)
		S0_list = np.linspace(pmin[2],pmax[2],Nword)
		L_array, Din_array, S0_array = np.meshgrid(L_list,Din_list,S0_list)
		L_array = L_array.flatten()
		Din_array = Din_array.flatten()
		S0_array = S0_array.flatten()
		Nmicro = L_array.size
		dict_pars = np.zeros((Nmicro,3))
		dict_pars[:,0] = L_array
		dict_pars[:,1] = Din_array
		dict_pars[:,2] = S0_array
	
	else:
		raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
	
	# Dictionary of MRI signals
	npars = dict_pars.shape[1]	
	dict_sigs = np.zeros((Nmicro,mripar.shape[1]))	
	for qq in range(0,Nmicro):      # Loop through different microstructures in the dictionary
		# Synthesise signals without relaxation-weighting and store them
		synsigs = _sig(mripar,dict_pars[qq,:],modstr)
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

		
		sliceinfo = [navg,mripar,dict_pars,dict_sigs,k_data_sl,m_data_sl,sgm_exist,sgm_data_sl,ww,pmin,pmax,modstr,nlinalgo,regnorm,regw]   
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

	
	if(modstr=='DinDexTD'):
		map_strings = ['Lum.nii','D0um2ms-1','fin.nii','Dexinfum2ms-1','Betaum2.nii','S0.nii']
	elif(modstr=='DinDex'):
		map_strings = ['Lum.nii','D0um2ms-1','fin.nii','Dexinfum2ms-1','S0.nii']
	elif(modstr=='Din'):
		map_strings = ['Lum.nii','D0um2ms-1','S0.nii']
	else:
		raise RuntimeError('ERROR: signal model {} is unknown.'.format(modstr))
		
	# Save parametric maps
	fin_notfound = True
	for uu in range(0,npars):
	
		# Keep track of cell size
		if(map_strings[uu]=='Lum.nii'):
			Lmap = np.squeeze(Tparmap[:,:,:,uu])
		
		# Keep track of intra-cellular tissue fraction
		if(map_strings[uu]=='fin.nii'):
			Finmap = np.squeeze(Tparmap[:,:,:,uu])
			fin_notfound = False

		# Rescale S0 estimate
		if(map_strings[uu]=='S0.nii'):
			Tparmap[:,:,:,uu] = k_data*Tparmap[:,:,:,uu]*np.nanmax(m_data,axis=3)
			S0fitted = 1.0*Tparmap[:,:,:,uu]
	
		niiout_obj = nib.Nifti1Image(np.squeeze(Tparmap[:,:,:,uu]),m_obj.affine,buffer_header)
		nib.save(niiout_obj, '{}_{}'.format(output,map_strings[uu]))
	
	## Check whether Fin can be estimated even for model Din, given a reference S0
	if(fin_notfound):

		# Check whether the user provided a reference S0 - if so, load it and check consistency
		if (s0file is not None):
			try:
				sref_obj = nib.load(s0file)
			except:
				print('')
				raise RuntimeError('ERROR: the 3D input file {} does not exist or is not in NIFTI format.'.format(s0file))
			sref_data = sref_obj.get_fdata()
			sref_size = sref_data.shape
			sref_size = np.array(sref_size)
			sref_header = sref_obj.header
			sref_affine = sref_header.get_best_affine()
			sref_dims = sref_obj.shape
			if sref_size.size!=3:
				print('')
				raise RuntimeError('ERROR: the 3D input file {} is not a 3D NIFTI.'.format(s0file))
			if ( (np.sum(sref_affine==m_affine)!=16) or (sref_dims[0]!=m_dims[0]) or (sref_dims[1]!=m_dims[1]) or (sref_dims[2]!=m_dims[2]) ):
				print('')
				raise RuntimeError('ERROR: the header geometry of {} and {} do not match.'.format(mrifile,s0file))

			Finmap = S0fitted/sref_data
			Finmap[Finmap<0.0] = 0.0
			Finmap[Finmap>1.0] = 1.0
			Finmap[np.isnan(Finmap)] = 0.0
			Finmap[np.isinf(Finmap)] = 0.0

		else:
			Finmap = np.ones((m_size[0],m_size[1],m_size[2]))

		niiout_obj = nib.Nifti1Image(Finmap,m_obj.affine,buffer_header)
		nib.save(niiout_obj, '{}_fin.nii'.format(output))


	# Calculate cellularity in cells/mm3 and save it
	cellmap = Finmap/( (0.001*Lmap)*(0.001*Lmap)*(0.001*Lmap) )
	cellularity_obj = nib.Nifti1Image(cellmap,m_obj.affine,buffer_header)
	nib.save(cellularity_obj, '{}_cellsmm-3.nii'.format(output))

	# Calculate cellularity in cells/mm2 and save it
	cellmap2D = Finmap/( (0.001*Lmap)*(0.001*Lmap) )
	cellularity2D_obj = nib.Nifti1Image(cellmap2D,m_obj.affine,buffer_header)
	nib.save(cellularity2D_obj, '{}_cellsmm-2.nii'.format(output))

		
	
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
	parser = argparse.ArgumentParser(description='This tool estimates the parameters of a 2-compartment model of restricted diffusion within spheres and extra-cellular hindered diffusion, via regularised non-linear optimisation of  a likelihood function under an offset-Gaussian noise model.Third-party dependencies: nibabel, numpy, scipy. Developed and validated with versions: nibabel 3.2.1, numpy 1.21.5, scipy 1.7.3. Author: Francesco Grussu, Vall d Hebron Institute of Oncology (VHIO). Email: <francegrussu@gmail.com> <fgrussu@vhio.net>.')
	parser.add_argument('s_file', help='path of a 4D NIFTI file storing M diffusion MRI measurements acquired at multiple b-values and diffusion times')
	parser.add_argument('scheme_file', help='path of a text file storing information on b-values and diffusion times  corresponding to each volume of s_file. The acquisition must be a standard pulsed gradient spin echo (PGSE, also known as single linear diffusion encoding). This file must contain a space-separated array of 3 x M elements, arranged along 3 lines  (first line: b-values in s/mm2; second line: gradient duration delta in ms; third line: gradient separation Delta in ms).')
	parser.add_argument('out', help='root file name of output files; output NIFTIs will be stored as double-precision floating point images  (FLOAT64), and the file names will end in  *_Lum.nii (cell diameter L in um),  *_D0um2ms-1.nii (intrinsic intra-cellular diffusivity D0 in um2/ms), *_Dexinfum2ms-1.nii (extra-cellular apparent diffusion coefficient parameter Dexinf in um2/ms), *_Betaum2.nii (extra-cellular apparent diffusion coefficient parameter Beta -- note that the  extra-cellular apparent diffusion coefficient Dex is written as Dex = Dexinf + Beta/t, where t is  the gradient separation of measurements kept after b-value thresholding (see input parameter bth),  *_fin.nii (intra-cellular tissue signal fraction Fin), *_S0.nii (non-DW signal level S0), *_cellsmm-2.nii (2D histology-like cellularity C in cells/mm2), *_cellsmm-3.nii (cellularity C in cells/mm3),  *_exit.nii (voxel-wise exit code; -1: warning, failure in non-linear fitting; 0: background; 1 successful parameter estimation). If a noise map was provided with the noisefile input parameter, additional output NIFTI files storing quality of fit metrics are stored, i.e.: *_logL.nii (log-likelihood), *_BIC.nii (Bayesian Information Criterion), and *_AIC.nii (Akaike Information Criterion). The number of parametric maps outputted depends on the model specified with input parameter modstr (see below). These will be:  L, D0, Fin, S0 for model "Din"; L, D0, Dexinf, Fin, S0 for model "DinDex"; L, D0, Dexinf, Beta, Fin, S0 for model "DinDexTD"')
	parser.add_argument('--mask', metavar='<file>', help='3D mask in NIFTI format (computation will be performed only in voxels where mask = 1)')
	parser.add_argument('--noise', metavar='<file>', help='3D noise standard deviation map in NIFTI format. If provided, the signal level will be compared to the expected Rician noise floor.')
	parser.add_argument('--s0ref', metavar='<file>', help='3D reference non-DW signal S0 in NIFTI format. Relevant only if the fitted model is "Din" (see modstr below). If provided, S0_Din, i.e., the non-DW signal level estimated via "Din" fitting, will be compared to S0 to estimate the intra-cellular signal fraction F as F = S0_Din/S0. F will be stored as a NIFTI map as for models "DinDex" and "DinDexTD"')
	parser.add_argument('--savg', metavar='<num>', default='1', help='number of signal averages used for MRI data acquisition (default: 1). This parameter is used for the estimation of the noise floor (it is ignored if the option --noise is not used). Note that in some vendors, this parameter is also referred to as number of excitations (NEX).')
	parser.add_argument('--nw', metavar='<num>', default='6', help='number of values to test for each unknown tissue parameter in the grid search (it must be an integer; default 10)')
	parser.add_argument('--pmin', metavar='<list>', help='list or array storing the lower bounds for tissue parameters. These are: L,D0,S0 for model "Din";  L,D0,F,Dexinf,S0 for model "DinDex"; L,D0,F,Dexinf,Beta,S0 for model "DinDexTD". The symbols stand for: L -> cell size (diameter), in (um); D0 -> intrinsic intra-cell diffusivity in (um2/ms);  F -> intra-cellular signal fraction; Dexinf -> long-time extra-cellular apparent diffusion coefficient, in (um2/ms); Beta -> extra-cellular diffusion-time dependence coefficient in (um2). Note that the extra-cellular apparent diffusion coefficient is written as Dex = Dexinf + Beta/t, where t is the gradient separation; S0 -> non-DW signal level, with respect to the maximum measured signal (parametrisation: S(b = 0) = S0*max(S)).  Default: "8.0,0.8,0.01" for model "Din"; "8.0,0.8,0.01,0.01,0.6" for model "DinDex"; "8.0,0.8,0.01,0.01,0.01,0.6" for model "DinDexTD". For more information on the models, please look at input parameter modstr below.')
	parser.add_argument('--pmax', metavar='<list>', help='list or array storing the upper bounds for tissue parameters. These are: L,D0,S0 for model "Din";  L,D0,F,Dexinf,S0 for model "DinDex"; L,D0,F,Dexinf,Beta,S0 for model "DinDexTD". The symbols stand for: L -> cell size (diameter), in (um); D0 -> intrinsic intra-cell diffusivity in (um2/ms);  F -> intra-cellular signal fraction; Dexinf -> long-time extra-cellular apparent diffusion coefficient, in (um2/ms); Beta -> extra-cellular diffusion-time dependence coefficient in (um2). Note that the extra-cellular apparent diffusion coefficient is written as Dex = Dexinf + Beta/t, where t is the gradient separation; S0 -> non-DW signal level, with respect to the maximum measured signal (parametrisation: S(b = 0) = S0*max(S)).  Default: "40.0,3.0,1.4" for model "Din"; "40.0,3.0,1.0,3.0,1.4" for model "DinDex"; "40.0,3.0,1.0,3.0,10.0,1.4" for model "DinDexTD". For more information on the models, please look at input parameter modstr below.')
	parser.add_argument('--reg', metavar='<Lnorm,weight>', help='comma-separated list of parameters for fitting regularisation specifying i) the type of L-norm (1 for LASSO, 2 for Tikhonov), ii) the weight of the regulariser, ranging in [0.0,1.0]. Default: 2,0.001 (L2 norm, with a weight of 0.001). Set 2,0.0 for a standard non-linear least square fitting with no regularisation.')
	parser.add_argument('--ncpu', metavar='<num>', default='1', help='number of threads to be used for computation (default: 1, single thread)')
	parser.add_argument('--sldim', metavar='<num>', default='2', help='image dimension along which parallel computation will be exectued when nthread > 1 (it can be 0, 1, 2; default 2, implying parallel processing along the 3rd image dimension)')
	parser.add_argument('--nlalgo', metavar='<string>', default='trust-constr', help='algorithm to be used for constrained objective function minimisation in non-linear fitting (relevant if --nlfit 1). Choose among: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", and "trust-constr" (default: "trust-constr" - see documentation of scipy.optimize.minimize for information on the optimisation algorithm)')
	parser.add_argument('--modstr', metavar='<string>', default='DinDex', help='string specifying the signal model to fit. Choose among: Din (extra-vascular signal dominated by intra-cellular diffusion), DinDex (extra-vascular signal features both intra-cellular and extra-cellular contributions, without diffusion time dependence in the extra-cellular ADC), DinDexTD (extra-vascular signal features both intra-cellular and extra-cellular contributions, with diffusion time dependence in the extra-cellular ADC). Default: DinDex. Intra-cellular diffusion is modelled using the Gaussian Phase Approximation (GPA) formula for diffusion within cylinders as in Van Geldern et al, "Evaluation of restricted diffusion in cylinders. Phosphocreatine in rabbit leg muscle", J Magn Reson B 1994, 103(3):255-60, doi: 10.1006/jmrb.1994.1038 ')
	args = parser.parse_args()


	### Get input arguments
	sfile = args.s_file
	schfile = args.scheme_file
	outroot = args.out
	maskfile = args.mask
	sgmfile = args.noise
	b0ref = args.s0ref
	nex = int(args.savg)
	nword = int(args.nw)
	cpucnt = int(args.ncpu)
	slptr = int(args.sldim)
	nlopt = args.nlalgo
	mymodel = args.modstr

	# Lower bounds
	if args.pmin is not None:
		pMin = (args.pmin).split(',')
		pMin = np.array( list(map( float, pMin )) )
	else:
		if(mymodel=='DinDexTD'):
			pMin = np.array([8.0,0.8,0.01,0.01,0.01,0.6])
		elif(mymodel=='DinDex'):
			pMin = np.array([8.0,0.8,0.01,0.01,0.6])
		elif(mymodel=='Din'):
			pMin = np.array([8.0,0.8,0.01])
		else:
			raise RuntimeError('ERROR: signal model {} is unknown.'.format(mymodel))
	
	# Upper bounds
	if args.pmax is not None:
		pMax = (args.pmax).split(',')
		pMax = np.array( list(map( float, pMax )) )
	else:
		if(mymodel=='DinDexTD'):
			pMax = np.array([40.0,3.0,1.0,3.0,10.0,1.4])
		elif(mymodel=='DinDex'):
			pMax = np.array([40.0,3.0,1.0,3.0,1.4])
		elif(mymodel=='Din'):
			pMax = np.array([40.0,3.0,1.4])
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
	print('                          pgse2cylperpinex.py                          ')
	print('***********************************************************************')
	print('')
	print('** 4D NIFTI file with MRI measurements: {}'.format(sfile))
	print('** MRI sequence parameter text file: {}'.format(schfile))
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
	if( b0ref is not None ):
		print('** Optional 3D NIFTI file with non-DW signal reference (used only for model "Din"): {}'.format(b0ref))
	print('** Output root name: {}'.format(outroot))
		

	### Run computation
	run(sfile, schfile, outroot, maskfile=maskfile, noisefile=sgmfile, s0file=b0ref, navg=nex, Nword=nword, pmin=pMin, pmax=pMax, regnorm=Lnorm, regw=Lweight, nthread=cpucnt, slicedim=slptr, nlinalgo=nlopt, modstr=mymodel)
	
	### Job done
	sys.exit(0)

